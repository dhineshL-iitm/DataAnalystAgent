import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 150))


def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        keys_list: list of keys in order
        type_map: dict key -> casting function
    """
    import re

    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float,
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map


@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif (
            any(url.lower().endswith(ext) for ext in (".xls", ".xlsx"))
            or "spreadsheetml" in ctype
        ):
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(
            r"/wiki/|\.org|\.com", url, re.IGNORECASE
        ):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = (
            df.columns.map(str).str.replace(r"\[.*\]", "", regex=True).str.strip()
        )

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first : i + 1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}


SCRAPE_FUNC = r"""
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
"""


def write_and_run_temp_python(
    code: str, injected_pickle: str = None, timeout: int = 60
) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    # create file content
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    # inject df if a pickle path provided
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        # ensure data exists so user code that references data won't break
        preamble.append("data = globals().get('data', {})\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    helper = r"""
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
"""

    # Build the code to write
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    # ensure results printed as json
    script_lines.append(
        "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout
        )
        if completed.returncode != 0:
            # collect stderr and stdout for debugging
            return {
                "status": "error",
                "message": completed.stderr.strip() or completed.stdout.strip(),
            }
        # parse stdout as json
        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not parse JSON output: {str(e)}",
                "raw": out,
            }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

tools = [scrape_url_to_dataframe]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)


def run_agent_safely(llm_input: str) -> Dict:
    """
    1. Run the agent_executor.invoke to get LLM output
    2. Extract JSON, get 'code' and 'questions'
    3. Detect scrape_url_to_dataframe("...") calls in code, run them here, pickle df and inject before exec
    4. Execute the code in a temp file and return results mapping questions -> answers
    """
    try:
        response = agent_executor.invoke(
            {"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS}
        )
        raw_out = (
            response.get("output")
            or response.get("final_output")
            or response.get("text")
            or ""
        )
        if not raw_out:
            return {"error": f"Agent returned no output. Full response: {response}"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if (
            not isinstance(parsed, dict)
            or "code" not in parsed
            or "questions" not in parsed
        ):
            return {"error": f"Invalid agent response format: {parsed}"}

        code = parsed["code"]
        questions: List[str] = parsed["questions"]

        # Detect scrape calls; find all URLs used in scrape_url_to_dataframe("URL")
        urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
        pickle_path = None
        if urls:
            # For now support only the first URL (agent may code multiple scrapes; you can extend this)
            url = urls[0]
            tool_resp = scrape_url_to_dataframe(url)
            if tool_resp.get("status") != "success":
                return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
            # create df and pickle it
            df = pd.DataFrame(tool_resp["data"])
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name
            # Make sure agent's code can reference df/data: we will inject the pickle loader in the temp script

        # Execute code in temp python script
        exec_result = write_and_run_temp_python(
            code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS
        )
        if exec_result.get("status") != "success":
            return {
                "error": f"Execution failed: {exec_result.get('message', exec_result)}",
                "raw": exec_result.get("raw"),
            }

        # exec_result['result'] should be results dict
        results_dict = exec_result.get("result", {})
        # Map to original questions (they asked to use exact question strings)
        output = {}
        for q in questions:
            output[q] = results_dict.get(q, "Answer not found")
        return output

    except Exception as e:
        logger.exception("run_agent_safely failed")
        return {"error": str(e)}


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke(
                {"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS}
            )
            raw_out = (
                response.get("output")
                or response.get("final_output")
                or response.get("text")
                or ""
            )
            print(raw_out)
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(
            code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS
        )
        if exec_result.get("status") != "success":
            return {
                "error": f"Execution failed: {exec_result.get('message')}",
                "raw": exec_result.get("raw"),
            }

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}
