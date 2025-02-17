import os
import yaml
import asyncio
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

load_dotenv()

# GraphRAG imports
import graphrag.api as api
from graphrag.cli.initialize import initialize_project_at
from graphrag.index.typing import PipelineRunResult
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_relationships,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from graphrag.config.init_content import INIT_DOTENV, INIT_YAML
from graphrag.prompts.index.claim_extraction import CLAIM_EXTRACTION_PROMPT
from graphrag.prompts.index.community_report import (
    COMMUNITY_REPORT_PROMPT,
)
from graphrag.prompts.index.entity_extraction import GRAPH_EXTRACTION_PROMPT
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import (
    REDUCE_SYSTEM_PROMPT,
)
from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT

from utils import create_env_file, file_check_list

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

async def run_command(root: str):
    """Run the indexing command as an async subprocess."""
    job_id = root
    running_jobs.add(job_id)
    print(f"Running job: {job_id}")

    try:
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "graphrag.index", "--root", root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout_output, stderr_output = await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr")
        )

        rc = await process.wait()
        print(f"Indexing finished with return code: {rc}")

        return {
            "status": "success" if rc == 0 else "error",
            "message": "Job completed",
            "stdout": stdout_output,
            "stderr": stderr_output
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
    """Endpoint to start the indexing job."""
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