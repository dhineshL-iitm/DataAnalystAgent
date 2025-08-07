from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import os
import asyncio

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])


async def llm(prompt: str):
    # set openai key in OPENAI_API_KEY
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(response)


@app.get("/")
async def root():
    return {"status": "Healthy"}


@app.post("/api")
async def process_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_name = file.filename
        return {
            "file_size": content,
            "file_name": file_name,
            "message": "File processed",
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


if __name__ == "__main__":
    asyncio.run(llm("Hi, count 1 to 10"))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
