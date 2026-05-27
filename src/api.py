import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from src.config import FaceBlurConfig, DetectorType, BlurType
from src.video_pipeline import VideoPipeline
from src.image_pipeline import ImagePipeline
from src.utils import setup_logging, is_video_file, is_image_file

app = FastAPI(title="Face Blur API", description="API for processing images and videos with face blurring.")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logging("INFO")

# Paths
ROOT_DIR = Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/api/process")
async def process_media(
    file: UploadFile = File(...),
    detector: str = Form("scrfd"),
    blur_type: str = Form("gaussian"),
    blur_strength: int = Form(99),
    confidence_threshold: float = Form(0.5)
):
    try:
        # Save uploaded file
        input_path = INPUT_DIR / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Received file: {file.filename}")

        # Configure pipeline
        config = FaceBlurConfig(
            detector=detector,
            blur_type=blur_type,
            blur_strength=blur_strength,
            confidence_threshold=confidence_threshold,
            # Optimizations for UI responsiveness
            frame_skip=2,
            temporal_smoothing=True
        )

        output_filename = f"blurred_{file.filename}"
        output_path = OUTPUT_DIR / output_filename
        
        # Route to appropriate pipeline
        if is_video_file(str(input_path)):
            pipeline = VideoPipeline(config)
            result_path = pipeline.process(str(input_path), str(output_path))
            media_type = "video"
        elif is_image_file(str(input_path)):
            pipeline = ImagePipeline(config)
            result_path = pipeline.process(str(input_path), str(output_path))
            media_type = "image"
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format. Please upload an image or video."}
            )

        return JSONResponse({
            "status": "success",
            "media_type": media_type,
            "filename": output_filename,
            "download_url": f"/api/download/{output_filename}"
        })

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path=file_path, filename=filename)

def run_server():
    """Start the FastAPI server."""
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    run_server()
