from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
from PIL import Image
import base64
import os

app = FastAPI(title="Strawberry Ripeness Detection API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = "best.pt"

@app.on_event("startup")
async def load_model():
    global model
    print("Loading YOLO model...")
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Make sure to train the model first.")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Strawberry Ripeness API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded yet"})
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference (lowered threshold since model is under-trained)
        results = model.predict(image, conf=0.05)
        
        # Parse results
        result = results[0]
        boxes = []
        for box in result.boxes:
            b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            c = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = result.names[c]
            
            boxes.append({
                "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                "class_id": c,
                "class_name": name,
                "confidence": conf
            })
            
        # Optional: return an image with drawn bounding boxes
        # We can either draw them on frontend, or return base64 image from backend.
        # Returning bounding box coords is generally better practice for a frontend to handle.
        
        return {
            "predictions": boxes,
            "image_width": image.width,
            "image_height": image.height
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
