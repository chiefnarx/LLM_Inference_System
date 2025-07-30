from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline, set_seed
import yaml
from datetime import datetime
import time
import uuid

app = FastAPI()

# Define the input format for /v1/completions
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0

# Load and store models in a dictionary
model_registry = {}


def load_models():
    with open("models.yaml", "r") as f:
        config = yaml.safe_load(f)

    for name, info in config["models"].items():
        model_id = info["id"]
        print(f"Loading model: {name} from {model_id}")
        model_registry[name] = pipeline(
            "text-generation",
            model=model_id,
            do_sample=True
        )


# Run model loading at startup
load_models()

@app.get("/")
def read_root():
    return {"message": "LLM Inference System is live!"}

@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    model_name = request.model

    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Here are the available models: {available}"
        )

    if not request.prompt or request.prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if request.max_tokens is None or request.max_tokens > 512:
        raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 512.")

    try:
        generator = model_registry[model_name]  # <-- This line was missing

        output = generator(
            request.prompt,
            max_length=request.max_tokens,
            temperature=request.temperature,
            num_return_sequences=1
        )

        generated_text = output[0]["generated_text"]


        # Fake token usage stats (we can improve this later)
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(generated_text.split()) - prompt_tokens
        total_tokens = prompt_tokens + completion_tokens

        return {
            "id": f"cmpl-{str(uuid.uuid4())[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
