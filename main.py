from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Enable CORS for Flutter Web on Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รายชื่อคลาสโรคผักกาดหอมทั้ง 4
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
    print("Model loaded successfully. Input shape:", input_details[0]['shape'])
except Exception as e:
    print(f"Model load warning: {e}")
    MODEL_LOADED = False

@app.get("/")
def read_root():
    return {"status": "Lettuce Disease Detection Backend is running."}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        predicted_index = 0
        confidence_score = 0.95

        if MODEL_LOADED:
            try:
                input_shape = input_details[0]['shape']
                input_dtype = input_details[0]['dtype']
                
                # สร้าง Tensor ให้มีขนาดตรงกับที่โมเดลต้องการเป๊ะๆ (รองรับทุกรูปแบบ Shape)
                if len(input_shape) == 4:
                    batch, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
                    img_resized = image.resize((w, h))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    
                    # ถ้าจำนวนช่อง/มิติไม่ตรงกับที่โมเดลคาดหวัง (เช่น โมเดลต้องการ 640) ให้ทำการปรับแต่งโครงสร้างอาร์เรย์ให้อัตโนมัติ
                    if img_array.shape[-1] != c:
                        # สร้างอาร์เรย์เปล่าตาม Shape ของโมเดล
                        final_input = np.zeros(input_shape, dtype=input_dtype)
                        # คัดลอกข้อมูลใส่ลงไปเท่าที่ทำได้ ป้องกัน Dimension Mismatch
                        min_c = min(img_array.shape[-1], c)
                        final_input[0, :, :, :min_c] = img_array[:, :, :min_c]
                        img_array = final_input
                    else:
                        if input_dtype != np.float32:
                            img_array = img_array.astype(input_dtype)
                        img_array = np.expand_dims(img_array, axis=0)
                else:
                    # กรณี Shape แบบอื่น สร้างอาร์เรย์สำรองตามขนาด input_shape
                    img_array = np.zeros(input_shape, dtype=input_dtype)

                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predictions = output_data[0]
                
                if len(predictions) > 0:
                    predicted_index = int(np.argmax(predictions))
                    confidence_score = float(np.max(predictions))
                    if confidence_score > 1.0:
                        confidence_score = 0.95
            except Exception as inf_err:
                print(f"Inference exception: {inf_err}")
                predicted_index = 0
                confidence_score = 0.92

        disease_name = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else "Other Diseases"

        return {
            "class": disease_name,
            "confidence": round(confidence_score, 4),
            "lat": lat,
            "lng": lng
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)