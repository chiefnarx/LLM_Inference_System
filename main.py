from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline
import yaml
from datetime import datetime
import time
import uuid
import torch
import traceback  # For full error logging
import numpy 

app = FastAPI()

# Input format for /v1/completions
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0

# Store models in memory
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
            detail=f"Model '{model_name}' not found. Available models: {available}"
        )

    if not request.prompt or request.prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if request.max_new_tokens is None or request.max_new_tokens > 512:
        raise HTTPException(status_code=400, detail="max_new_tokens must be between 1 and 512.")

    try:
        generator = model_registry[model_name]

        output = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            pad_token_id=50256,
            num_return_sequences=1
        )

        try:
            generated_text = output[0]["generated_text"]
        except KeyError:
            generated_ids = output[0].get("generated_token_ids")
            if generated_ids is not None:
                if torch.is_tensor(generated_ids):
                    generated_ids = generated_ids.tolist()
                generated_text = generator.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                raise HTTPException(status_code=500, detail="Model output format not recognized.")

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
        print("Error during completion:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{str(e)}")
