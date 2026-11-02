
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from huggingface_hub import snapshot_download


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Smaug AI Image Detector...")

    model_path = snapshot_download(
        "abacusai/Smaug-AI-Image-Detector"
    )

    sys.path.insert(0, model_path)

    from smaug_image_detect import SmaugImageDetect

    app.state.detector = SmaugImageDetect.from_pretrained(model_path)

    print("Smaug detector loaded successfully!")

    yield

    print("TrueLens AI shutting down...")


app = FastAPI(
    title="TrueLens AI",
    description="Multimodal Deepfake and Disinformation Detection Platform",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "Welcome to TrueLens AI",
        "status": "running"
    }


@app.post("/api/detect/image")
def detect_image(file: UploadFile = File(...)):

    contents = file.file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty"
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB"
        )

    try:
        image = Image.open(BytesIO(contents))
        image.verify()

        image = Image.open(BytesIO(contents))

        if image.format not in ["JPEG", "PNG", "WEBP"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG, PNG and WEBP images are supported"
            )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )

    extension = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp"
    }[image.format]

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        score = app.state.detector(temp_path)

        return {
            "filename": file.filename,
            "ai_generated_score": float(score),
            "score_scale": "0 to 1",
            "note": "Model score is an estimate, not proof of authenticity."
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image detection failed: {str(error)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
