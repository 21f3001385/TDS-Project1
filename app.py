from fastapi import FastAPI, HTTPException
from typing import Optional
import os
import json
import subprocess
import datetime
import sqlite3
from pathlib import Path
import aiohttp
import asyncio
from PIL import Image

app = FastAPI()

async def call_llm(prompt: str) -> str:
    """Make a call to GPT-4-mini via AI Proxy."""
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="AIPROXY_TOKEN not set")
        
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.aiproxy.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail="LLM API call failed")
            data = await response.json()
            return data["choices"][0]["message"]["content"]

async def parse_task(task_description: str) -> dict:
    """Parse the task description using LLM to identify the operation and parameters."""
    prompt = f"""Parse this task and identify:
1. The type of operation (e.g., file read, format, count, sort)
2. The input file path
3. The output file path
4. Any specific parameters or requirements

Task: {task_description}

Return as JSON with these keys: operation, input_path, output_path, parameters"""

    response = await call_llm(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse task")

async def execute_task(task_info: dict) -> None:
    """Execute the parsed task."""
    # Ensure we're only accessing files within /data
    if not task_info["input_path"].startswith("/data/") or not task_info["output_path"].startswith("/data/"):
        raise HTTPException(status_code=400, detail="Can only access files within /data directory")

    operation = task_info["operation"]
    
    if operation == "count_weekday":
        with open(task_info["input_path"], 'r') as f:
            dates = f.readlines()
        count = sum(1 for date in dates if datetime.datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == task_info["parameters"]["weekday"])
        with open(task_info["output_path"], 'w') as f:
            f.write(str(count))
            
    elif operation == "format_markdown":
        subprocess.run(["npx", "prettier@3.4.2", "--write", task_info["input_path"]])
        
    elif operation == "sort_json":
        with open(task_info["input_path"], 'r') as f:
            data = json.load(f)
        sorted_data = sorted(data, key=lambda x: (x["last_name"], x["first_name"]))
        with open(task_info["output_path"], 'w') as f:
            json.dump(sorted_data, f, indent=2)
            
    elif operation == "extract_email":
        with open(task_info["input_path"], 'r') as f:
            email_content = f.read()
        prompt = f"Extract just the sender's email address from this email:\n\n{email_content}"
        email = await call_llm(prompt)
        with open(task_info["output_path"], 'w') as f:
            f.write(email.strip())
            
    # Add more operations as needed...

@app.post("/run")
async def run_task(task: str):
    try:
        task_info = await parse_task(task)
        await execute_task(task_info)
        return {"status": "success"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Can only read files from /data directory")
        
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)