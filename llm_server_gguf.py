# llm_server_gguf.py
# Start the local GGUF model server (Llama 3.1 8B) for Power BI integration

#--------------------------------------------------
# ORIGINAL CODE
#--------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List
from contextlib import asynccontextmanager
import logging, threading, json, re, os, uvicorn

# simple in-memory cache
_cache = {}
def cache_get(key):
    return _cache.get(key)

def cache_set(key, value):
    _cache[key] = value



# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = r"E:\AJAY\powerbi_models\llama-3.1-8B-instruct-Q5_K_M.gguf"
N_THREADS = 8          # adjust for your CPU
N_GPU_LAYERS = 20      # how many layers to load on GPU (~80% GPU use)
N_CTX = 2048
HOST = "127.0.0.1"
PORT = 7860
# ===========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("llm_server")

# Load model once (outside Power BI)
log.info(f"Loading model: {MODEL_PATH}")
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=N_GPU_LAYERS)
lock = threading.Lock()
log.info("✅ Model loaded successfully.")

app = FastAPI(title="PowerBI Llama GGUF Server")

class RequestItem(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0

class BatchRequest(BaseModel):
    items: List[RequestItem]

@app.post("/batch_generate")
def batch_generate(req: BatchRequest):
    responses = []
    for item in req.items:
        prompt = item.prompt
        with lock:
            out = llm(prompt, max_tokens=item.max_tokens, temperature=item.temperature)
        raw = out["choices"][0]["text"]

        # Try to parse JSON-like output
        suggestion = insight = drawback = ""
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                text = match.group(0).replace("'", '"')
                data = json.loads(text)
                suggestion = data.get("suggestion", "")
                insight = data.get("insight", "")
                drawback = data.get("drawback", "")
            else:
                lines = [x.strip() for x in raw.split("\n") if x.strip()]
                suggestion = lines[0] if len(lines) > 0 else ""
                insight = lines[1] if len(lines) > 1 else ""
                drawback = lines[2] if len(lines) > 2 else ""
        except Exception:
            pass

        responses.append({
            "suggestion": suggestion,
            "insight": insight,
            "drawback": drawback,
            "raw": raw
        })
    return {"results": responses}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run("llm_server_gguf:app", host=HOST, port=PORT, workers=1)
