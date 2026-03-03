from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from ai.ap_debug_render import render_one as render_ap
from ai.lat_debug_render import render_one as render_lat
from ultralytics import YOLO

# ------------------------------------------------------------
# Load YOLO model once at import time (production-safe)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"best.pt not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)


# ============================================================
# App Setup
# ============================================================

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATE_DIR)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Allow future flexibility (Render-safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Page Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ap", response_class=HTMLResponse)
async def ap_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# ============================================================
# Upload Helper
# ============================================================

def save_upload(file: UploadFile):
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return filename, filepath


# ============================================================
# AP Endpoint
# ============================================================

@app.post("/predict/ap")
async def predict_ap(file: UploadFile = File(...)):
    try:
        filename, filepath = save_upload(file)
        output_filename = f"ap_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        record = render_ap(
            image_path=filepath,
            label_path=None,
            out_path=output_path,
            debug=False
        )

        return JSONResponse({
            "success": True,
            "output_image": f"/outputs/{output_filename}",
            "meta": record
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


# ============================================================
# LAT Endpoint (future)
# ============================================================

@app.post("/predict/lat")
async def predict_lat(file: UploadFile = File(...)):
    try:
        filename, filepath = save_upload(file)
        output_filename = f"lat_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        record = render_lat(
            image_path=filepath,
            label_path=None,
            out_path=output_path,
            debug=False
        )

        return JSONResponse({
            "success": True,
            "output_image": f"/outputs/{output_filename}",
            "meta": record
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )