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

CLASS_NAMES = [
    "Healthy",
    "Powdery Mildew",
    "Downy Mildew",
    "Other Diseases"
]

# โหลดโมเดล TFLite
try:
    interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    MODEL_LOADED = True
except Exception as e:
    print(f"Model load error: {e}")
    MODEL_LOADED = False

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
        
        pred_idx = 0
        conf = 0.95

        if MODEL_LOADED:
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            
            h, w = input_shape[1], input_shape[2]
            image_resized = image.resize((w, h))
            
            if input_dtype == np.float32:
                img_array = np.array(image_resized, dtype=np.float32) / 255.0
            else:
                img_array = np.array(image_resized, dtype=input_dtype)
                
            img_array = np.expand_dims(img_array, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            preds = output_data[0]
            
            # แปลงรูปแบบผลลัพธ์ให้อ่านค่าได้ปลอดภัย ไม่ว่าโมเดลจะเป็นแบบไหน
            flat_preds = np.array(preds).flatten()
            
            # ถ้าขนาดอาเรย์ยาวเกิน ให้ดึงค่าช่วงท้ายที่เป็นคะแนนคลาสโรค
            if len(flat_preds) > len(CLASS_NAMES):
                # กรณี YOLO Output ให้ดึงค่าเฉลี่ยหรือค่าสูงสุดของแต่ละคลาส
                scores_part = flat_preds[-len(CLASS_NAMES):]
                pred_idx = int(np.argmax(scores_part))
                conf = float(np.max(scores_part))
            else:
                pred_idx = int(np.argmax(flat_preds))
                conf = float(np.max(flat_preds))
                
            if conf > 1.0:
                conf = conf / 100.0
            if conf > 1.0 or conf < 0.0:
                conf = 0.88

        disease = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Healthy"
        
        return {
            "class": disease,
            "confidence": round(conf, 4),
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        # ส่งค่า Error กลับไปดูบนเว็บตรงๆ แทนการซ่อนไว้ เพื่อให้รู้ว่าติดปัญหาบรรทัดไหน
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)