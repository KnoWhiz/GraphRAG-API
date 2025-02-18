import os
import yaml
import asyncio
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

load_dotenv()

from utils import create_env_file, file_check_list
from graphrag_doc_processor import generate_GraphRAG_embedding
from graphrag_get_response import get_GraphRAG_global_response

GRAPHRAG_API_KEY = os.getenv("GRAPHRAG_API_KEY")
GRAPHRAG_LLM_MODEL = os.getenv("GRAPHRAG_LLM_MODEL")
GRAPHRAG_API_BASE = os.getenv("GRAPHRAG_API_BASE")
GRAPHRAG_API_VERSION = os.getenv("GRAPHRAG_API_VERSION")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for FastAPI lifespan, to start the background job processor."""
    asyncio.create_task(background_job_processor())
    yield

app = FastAPI(lifespan=lifespan)
job_queue = asyncio.Queue()
running_jobs = set()
queue_lock = asyncio.Lock()

async def read_stream(stream, prefix):
    """Read and print lines from an async stream."""
    output = []
    async for line in stream:
        decoded_line = line.decode().strip()
        output.append(decoded_line)
        print(f"{prefix}: {decoded_line}")
    return '\n'.join(output)

async def run_command(root: str, file_path: str):
    """Run the indexing command using generate_GraphRAG_embedding."""
    job_id = root
    running_jobs.add(job_id)
    print(f"Running job: {job_id}")

    try:
        # Call generate_GraphRAG_embedding directly
        await generate_GraphRAG_embedding(root)
        
        return {
            "status": "success",
            "message": "Job completed successfully",
            "stdout": "GraphRAG embedding generation completed",
            "stderr": ""
        }
    except Exception as e:
        print(f"Error in run_command: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        running_jobs.remove(job_id)
        await process_next_job()

async def process_next_job():
    """Process the next job in the queue."""
    if not job_queue.empty():
        next_root = await job_queue.get()
        asyncio.create_task(run_command(next_root))

async def background_job_processor():
    """Continuously process jobs from the queue."""
    while True:
        if not job_queue.empty() and len(running_jobs) == 0:
            await process_next_job()
        await asyncio.sleep(1)

@app.get("/run-index")
async def run_index(root: Optional[str] = "."):
    # root = f"./graphrag_embedded_content/{course_id}/GraphRAG/"
    """Endpoint to start the indexing job using generate_GraphRAG_embedding."""
    print(f"Received request for /run-index - root: {root}")

    async with queue_lock:
        if root in running_jobs:
            return {"status": "running", "message": "Job is currently running.", "queueSize": job_queue.qsize()}

        await job_queue.put(root)
        if len(running_jobs) == 0:
            asyncio.create_task(process_next_job())
            return {"status": "running", "message": "Job started.", "queueSize": job_queue.qsize()}
        else:
            return {"status": "queued", "message": "Job queued.", "queueSize": job_queue.qsize()}

@app.get("/query")
async def query(query: str):
    """Endpoint to run a query command."""
    print(f"Received request for /query - query: {query}")

    try:
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "graphrag.query", "--root", ".", "--method", "global", query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout_output, stderr_output = await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr")
        )

        rc = await process.wait()
        return {
            "status": "success" if rc == 0 else "error",
            "message": "Query completed",
            "stdout": stdout_output,
            "stderr": stderr_output
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/status")
async def status():
    """Endpoint to get the status of the job queue and running jobs."""
    queue_size = job_queue.qsize()
    running_status = list(running_jobs) if running_jobs else "No jobs running"
    return {"queueSize": queue_size, "runningJobs": running_status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)