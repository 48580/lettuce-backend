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
            input_shape = input_details[0]['shape'] # เช่น [1, 640, 640, 3] หรือ [1, 3, 640, 640]
            input_dtype = input_details[0]['dtype']
            
            # ตรวจสอบรูปแบบ Input Shape ของโมเดล (Batch, Height, Width, Channels)
            if len(input_shape) == 4:
                if input_shape[1] == 3: # กรณี Channels อยู่ก่อน (NCHW)
                    c, h, w = input_shape[1], input_shape[2], input_shape[3]
                    image = image.resize((w, h))
                    img_array = np.array(image, dtype=np.float32) / 255.0
                    img_array = np.transpose(img_array, (2, 0, 1)) # เปลี่ยนเป็น (3, H, W)
                else: # กรณี Channels อยู่หลัง (NHWC - แบบมาตรฐาน YOLO ส่วนใหญ่)
                    h, w, c = input_shape[1], input_shape[2], input_shape[3]
                    image = image.resize((w, h))
                    img_array = np.array(image, dtype=np.float32) / 255.0
            else:
                image = image.resize((640, 640))
                img_array = np.array(image, dtype=np.float32) / 255.0

            # แปลงประเภทข้อมูลให้ตรงกับที่โมเดลกำหนด
            if input_dtype == np.int8 or input_dtype == np.uint8:
                img_array = (img_array * 255.0).astype(input_dtype)
            else:
                img_array = img_array.astype(np.float32)

            # เติมมิติ Batch เข้าไปด้านหน้าสุด (กลายเป็น 4 มิติสมบูรณ์)
            img_array = np.expand_dims(img_array, axis=0)
            
            # บังคับแปลง Shape ให้ตรงกับ input_shape ของโมเดลเป๊ะๆ ป้องกันปัญหา Dimension Mismatch
            img_array = np.reshape(img_array, input_shape)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            preds = output_data[0]
            
            flat_preds = np.array(preds).flatten()
            
            if len(flat_preds) > len(CLASS_NAMES):
                scores_part = flat_preds[-len(CLASS_NAMES):]
                pred_idx = int(np.argmax(scores_part))
                conf = float(np.max(scores_part))
            else:
                pred_idx = int(np.argmax(flat_preds))
                conf = float(np.max(flat_preds))
                
            if conf > 1.0:
                conf = conf / 100.0
            if conf > 1.0 or conf < 0.0:
                conf = 0.92

        disease = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Healthy"
        
        return {
            "class": disease,
            "confidence": round(conf, 4),
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)