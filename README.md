# 🍓 Strawberry Ripeness Detection System

![Strawberry Ripeness](https://img.shields.io/badge/Status-Completed-success)
![YOLOv8](https://img.shields.io/badge/AI_Model-YOLOv8-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)

Proyek ini adalah sistem deteksi tingkat kematangan buah strawberry menggunakan pendekatan **Computer Vision (AI)**. Sistem mendeteksi kematangan berdasarkan tiga kategori (kelas):
- **Fullripe** (Matang Penuh)
- **Semiripe** (Setengah Matang)
- **Unripe** (Belum Matang)

Proyek ini dibangun menggunakan arsitektur pemisahan *backend* (API untuk Machine Learning) dan *frontend* (Web Interface untuk end-user) guna memastikan performa tinggi dan skalabilitas.

---

## 🛠 Teknologi yang Digunakan

1. **AI & Machine Learning (Model)**
   - **YOLOv8 (Ultralytics)**: Model *state-of-the-art* untuk deteksi objek (Object Detection) secara cepat dan akurat.
   - **Python 3.11**: Bahasa pemrograman utama pemrosesan data & ML.
2. **Backend Server**
   - **FastAPI**: Framework web berkinerja tinggi untuk membangun API dengan Python.
   - **Uvicorn**: Server ASGI yang digunakan untuk menjalankan FastAPI.
3. **Frontend Web (User Interface)**
   - **Next.js 15 (App Router)**: Framework React fullstack untuk performa web modern.
   - **Tailwind CSS & shadcn/ui**: *Styling* modern dengan *Glassmorphism* dan komponen UI interaktif yang premium.

---

## 🚀 Panduan Setup & Menjalankan Aplikasi

Aplikasi ini dibagi menjadi 2 *service* utama: Backend (FastAPI) dan Frontend (Next.js). Anda perlu menjalankan keduanya agar aplikasi bisa terintegrasi (_end-to-end_).

### 1. Menjalankan Backend (FastAPI + YOLOv8)

Pastikan Anda berada di *root directory* proyek (`tomato-ripe`). Backend bertugas me-load model `best.pt` dan memproses gambar.

```bash
# 1. Buat Virtual Environment (disarankan menggunakan Python 3.11)
python3.11 -m venv venv

# 2. Aktivasi Virtual Environment
source venv/bin/activate  # Untuk Mac/Linux
# venv\Scripts\activate   # Untuk Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Jalankan Server FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*API sekarang berjalan di `http://localhost:8000`. Cek `http://localhost:8000/docs` untuk Swagger UI.*

### 2. Menjalankan Frontend (Next.js)

Buka terminal baru, lalu masuk ke folder `frontend`.

```bash
# 1. Masuk ke folder frontend
cd frontend

# 2. Install Node dependencies
npm install

# 3. Jalankan server Development
npm run dev
```
*Aplikasi Web Interface akan berjalan di `http://localhost:3000`.*

### 3. Cara Kerja Integrasi
- User mengunggah gambar melalui UI di `http://localhost:3000`.
- Gambar tersebut dikirim menggunakan AJAX (Fetch API) ke *endpoint* `POST http://localhost:8000/predict`.
- Backend mengeksekusi model YOLOv8 (file `best.pt`), menemukan *bounding boxes*, dan mengembalikan koordinat X/Y, label kelas, serta persentase *confidence* ke dalam format JSON.
- Frontend membaca data JSON tersebut, lalu menggambar matriks *bounding box* di atas gambar secara interaktif (Warna Merah/Oranye/Hijau).

---

## 📊 Proses Evaluasi & Metrik Model

Evaluasi model pada penelitian ini tidak dilakukan secara manual, melainkan otomatis di-generate oleh mesin YOLOv8 selama siklus *training* dan *validation*.

### Dari mana hasil evaluasi didapatkan?
Saat Anda menjalankan perintah training (`model.train()`), YOLOv8 otomatis menyisihkan gambar sebagai data validasi (Validation Set). Setiap selesai 1 *epoch*, YOLOv8 mengevaluasi prediksi dari model sementara terhadap *ground truth* (label asli).

Hasil evaluasi (metrik dan visualisasi) disimpan otomatis di dalam folder hasil training (misalnya: `runs/detect/runs/detect/strawberry_ripeness-3/`).

### Metrik yang Digunakan
Sistem ini dievaluasi berdasarkan metrik *Object Detection* standar:
1. **mAP50 (Mean Average Precision):** Mengukur akurasi deteksi secara keseluruhan pada ambang batas IoU (Intersection over Union) 0.5.
2. **Precision (Presisi):** Berapa persentase dari prediksi model yang *benar-benar akurat*? (Menekan *False Positives*).
3. **Recall (Sensitivitas):** Seberapa baik model dapat menemukan seluruh objek yang *seharusnya* ada? (Menekan *False Negatives*).
4. **Confusion Matrix:** Grafik heatmap multikelas yang menunjukkan jumlah prediksi benar vs salah (misalnya, berapa banyak "Semiripe" yang salah ditebak sebagai "Unripe").

Untuk melihat semua grafik evaluasi, cukup buka folder konfigurasi YOLO `runs/detect/...` di *root directory* proyek Anda. Di dalamnya, Anda akan menemukan file penting seperti:
- `results.png` (Loss & Precision-Recall curves)
- `confusion_matrix.png`
- `val_batch0_pred.jpg` (Visualisasi prediksi dataset validasi)
