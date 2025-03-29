from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from tryoff_inference import run_inference
from PIL import Image
import io
import requests
from typing import Optional
import uuid
import torch
from diffusers import FluxTransformer2DModel, FluxFillPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CATVTON-FLUX Tryoff API",
    description="API for virtual tryoff using CATVTON-FLUX model",
    version="1.0.0"
)

# Statebin configuration
STATEBIN_BASE_URL = "https://statebin.io"
STATEBIN_BUCKET = "catvton-flux"

# Global variables for models
transformer = None
pipe = None

async def load_models():
    """Load models at startup"""
    global transformer, pipe
    try:
        logger.info("Loading cat-tryoff-flux model...")
        transformer = FluxTransformer2DModel.from_pretrained(
            "xiaozaa/cat-tryoff-flux",
            torch_dtype=torch.bfloat16
        )
        logger.info("Loading FLUX.1-dev model...")
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    await load_models()

async def upload_to_statebin(file_path: str, file_name: str) -> str:
    """Upload a file to Statebin and return its URL"""
    api_key = os.getenv("STATEBIN_API_KEY")
    url = f"{STATEBIN_BASE_URL}/{STATEBIN_BUCKET}/{file_name}"
    headers = {"X-API-KEY": api_key}
    
    with open(file_path, 'rb') as f:
        response = requests.put(url, headers=headers, data=f)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to upload to Statebin: {response.text}")
    
    return url

@app.post("/tryoff")
async def tryoff(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    steps: int = 50,
    guidance_scale: float = 30.0,
    seed: int = 42,
    width: int = 576,
    height: int = 768,
):
    """
    Process a tryoff request with the given image and mask.
    Returns URLs to both the restored garment and tryon result.
    """
    if transformer is None or pipe is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Please try again in a few moments.")
        
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate unique IDs for this request
            request_id = str(uuid.uuid4())
            
            # Save uploaded files
            image_path = os.path.join(tmp_dir, "image.png")
            mask_path = os.path.join(tmp_dir, "mask.png")
            
            # Process and save image
            image_content = await image.read()
            image_pil = Image.open(io.BytesIO(image_content))
            image_pil.save(image_path)
            
            # Process and save mask
            mask_content = await mask.read()
            mask_pil = Image.open(io.BytesIO(mask_content))
            mask_pil.save(mask_path)
            
            # Run inference
            garment_result, tryon_result = run_inference(
                image_path=image_path,
                mask_path=mask_path,
                num_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                size=(width, height),
                pipe=pipe  # Pass the loaded pipeline
            )
            
            # Save results to temporary files
            garment_path = os.path.join(tmp_dir, f"garment_{request_id}.png")
            tryon_path = os.path.join(tmp_dir, f"tryon_{request_id}.png")
            
            garment_result.save(garment_path)
            tryon_result.save(tryon_path)
            
            # Upload results to Statebin
            garment_url = await upload_to_statebin(
                garment_path,
                f"garments/{request_id}.png"
            )
            
            tryon_url = await upload_to_statebin(
                tryon_path,
                f"tryons/{request_id}.png"
            )
            
            return JSONResponse({
                "garment_url": garment_url,
                "tryon_url": tryon_url,
                "request_id": request_id
            })
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if transformer is not None and pipe is not None else "loading",
        "models_loaded": transformer is not None and pipe is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)