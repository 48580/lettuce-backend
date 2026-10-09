from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# เปิดอนุญาต CORS ให้เว็บ Flutter บน Netlify สามารถส่ง request มาได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # อนุญาตทุกโดเมน (รวมถึง Netlify)
    allow_credentials=True,
    allow_methods=["*"],      # อนุญาตทุก HTTP Methods (POST, GET ฯลฯ)
    allow_headers=["*"],      # อนุญาตทุก Headers
)

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: str = Form("0.0"),
    lng: str = Form("0.0")
):
    # อ่านไฟล์รูปภาพที่ส่งมาจาก Flutter Web
    image_bytes = await file.read()
    
    # -------------------------------------------------------------
    # TODO: นำ image_bytes ไปเข้าโมเดลประมวลผลโรคผักของคุณตรงนี้
    # ตัวอย่างโค้ดจำลองผลลัพธ์ (กรุณาปรับให้ตรงกับโค้ดโมเดลเดิมของคุณ):
    # -------------------------------------------------------------
    disease_name = "Bacterial Soft Rot"  # ตัวอย่างชื่อโรค
    confidence_score = 0.95              # ตัวอย่างค่าความมั่นใจ (95%)

    return {
        "class": disease_name,
        "confidence": confidence_score,
        "lat": lat,
        "lng": lng
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)