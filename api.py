from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from torchvision.transforms.functional import to_tensor
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from pydantic import BaseModel, Field, validator
from typing import List
import torch
from pydantic import BaseModel 
import gc
import controlnet_hinter
import logging
import base64
import numpy as np
import traceback
import matplotlib.pyplot as plt

app = FastAPI()

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateImagesRequest(BaseModel):
    prompt: str
    logo_image_url: str
    controlnet_type: str
    num_steps: int = Query(20, description="Number of steps for diffusion")
    negative_prompt: str = ''  

    @validator('negative_prompt')
    def validate_negative_prompt(cls, value):
        if not value:
            raise ValueError("Negative prompt cannot be empty")
        return value

CONTROLNET_MAPPING = {
    "canny_edge": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "hinter": controlnet_hinter.hint_canny
    },
    "pose": {
        "model_id": "lllyasviel/sd-controlnet-openpose",
        "hinter": controlnet_hinter.hint_openpose
    },
    "depth": {
        "model_id": "lllyasviel/sd-controlnet-depth",
        "hinter": controlnet_hinter.hint_depth
    },
    "scribble": {
        "model_id": "lllyasviel/sd-controlnet-scribble",
        "hinter": controlnet_hinter.hint_scribble,
    },
    "segmentation": {
        "model_id": "lllyasviel/sd-controlnet-seg",
        "hinter": controlnet_hinter.hint_segmentation,
    },
    "normal": {
        "model_id": "lllyasviel/sd-controlnet-normal",
        "hinter": controlnet_hinter.hint_normal,
    },
    "hed": {
        "model_id": "lllyasviel/sd-controlnet-hed",
        "hinter": controlnet_hinter.hint_hed,
    },
    "hough": {
        "model_id": "lllyasviel/sd-controlnet-mlsd",
        "hinter": controlnet_hinter.hint_hough,
    }
}

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def generate_image(prompt, negative_prompt, controlnet_type, control_image, num_steps=20):
    device = torch.device("mps")
    torch.mps.empty_cache()
    gc.collect()

    base_model_path = "digiplay/Juggernaut_final"

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MAPPING[controlnet_type]["model_id"]).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path,
                                                            controlnet=controlnet).to(device)

    no_of_steps = num_steps
    guidace_scale = 7.0
    controlnet_conditioning_scale = 1.0

    my_images = pipe(
        prompt=[prompt] * 4,
        negative_prompt=[negative_prompt] * 4,
        width=256,
        height=256,
        image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=no_of_steps,
        guidance_scale=guidace_scale,
    )

    generated_images = my_images.images
    rows, cols = 2, 2
    num_images = min(4, len(generated_images))
    grid = image_grid(generated_images[:num_images], rows, cols)
    grid = grid.resize((512, 512))

    # Convert the generated image grid to base64-encoded string
    img_bytes = BytesIO()
    grid.save(img_bytes, format="PNG")
    img_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    return img_bytes.getvalue()
    
@app.post("/generate_images")
async def generate_images(request_data: GenerateImagesRequest):
    try:
        # Print received payload for debugging
        print("Received payload:", request_data.dict())

        # Extract relevant data from the request_data
        prompt = request_data.prompt
        negative_prompt = request_data.negative_prompt if request_data.negative_prompt else ""
        logo_image_url = request_data.logo_image_url
        controlnet_type = request_data.controlnet_type
        num_steps = request_data.num_steps
        
        logo_image = load_image(logo_image_url)
        control_image = CONTROLNET_MAPPING[controlnet_type]["hinter"](logo_image)

        if negative_prompt is not None:
            generated_image_bytes = generate_image(prompt, negative_prompt, controlnet_type, control_image, num_steps)
        else:
        # Handle the case where negative_prompt is empty
            return JSONResponse(content={"message": "Negative prompt is empty"}, status_code=422)
        
        return StreamingResponse(content=BytesIO(generated_image_bytes), media_type="image/png")
        

    except HTTPException as e:
        print("Validation error details:", e.detail)
        raise HTTPException(status_code=500, detail="Internal Server Error")