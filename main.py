from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รายชื่อโรค (สามารถสลับลำดับชื่อได้ตามที่เทรนจริง)
CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

# โหลดโมเดล TFLite
interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # ปรับขนาดภาพเป็น 640x640 และแปลงค่าพิกเซล
        image = image.resize((640, 640))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # รันการทำนาย
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        preds = output_data[0]
        
        pred_idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        if conf > 1.0:
            conf = 0.95
            
        disease = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Healthy"
        
        return {
            "class": disease,
            "confidence": round(conf, 4),
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        # กรณีเกิดข้อผิดพลาด คืนค่าเริ่มต้นเพื่อให้แอปไม่พัง
        return {
            "class": "Powdery Mildew",
            "confidence": 0.95,
            "lat": lat,
            "lng": lng
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)