# 🚢 Ship and Maritime Monitoring System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-FF4B4B.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

## 📖 Overview
The Ship and Maritime Monitoring System is a production-ready application designed to analyze Synthetic Aperture Radar (SAR) and optical imagery for maritime vessel detection. 

The system leverages a **YOLOv8** deep learning model to accurately detect ships, estimate their sizes, and classify them by risk and type. The application is split into a highly performant **FastAPI microservice** backend and a rich **Streamlit dashboard** frontend, orchestrated seamlessly using **Docker**.

---

## ✨ Key Features
- **Deep Learning Object Detection:** Powered by Ultralytics YOLOv8 for rapid and accurate ship identification.
- **Microservices Architecture:** 
  - **Backend:** A RESTful FastAPI service handling heavy ML inference and feature extraction.
  - **Frontend:** A beautiful, interactive Streamlit UI for visual analytics, risk distribution, and reporting.
- **Production-Ready:** Pre-loaded ML models on startup (via FastAPI lifespan), centralized structured logging, and robust error handling.
- **Exportable Analytics:** Download annotated images and detailed CSV reports of all detected vessels directly from the dashboard.

---

## 🏗️ Architecture

1. **`main_api.py` (Backend):** Exposes the `/analyze` POST endpoint. It decodes the uploaded image, runs the YOLOv8 model via `pipeline.py`, extracts vessel features (size, color heuristics, confidence), and returns a structured JSON payload.
2. **`app.py` (Frontend):** Consumes the backend API. Provides an intuitive UI for users to upload images, adjust confidence thresholds, and view comprehensive maritime traffic data.
3. **`pipeline.py` (Core):** Houses the ML orchestration, bounding box logic, classification heuristics, and image manipulation utilities.

---

## 🚀 Getting Started

There are two ways to run this application: via **Docker** (recommended for production) or **Locally via Virtual Environment** (for development).

### Method 1: Docker (Recommended)
You only need [Docker Desktop](https://docs.docker.com/desktop/) installed.

```bash
# Clone the repository and navigate to the project directory
docker compose up --build -d
```
- The **Frontend Dashboard** will be available at: `http://localhost:8501`
- The **Backend API** will be available at: `http://localhost:8000`

### Method 2: Local Native Development
If you prefer running it natively without Docker, ensure you have Python 3.11+ installed.

**1. Create and activate a virtual environment:**
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / Mac
python -m venv .venv
source .venv/bin/activate
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Backend API (Terminal 1):**
```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000
```

**4. Run the Frontend Dashboard (Terminal 2):**
*Open a new terminal window, activate the virtual environment, and run:*
```bash
streamlit run app.py
```

---

## 📡 API Reference
**Endpoint:** `POST /analyze`
- **Query Parameter:** `min_confidence` (float, default=0.50)
- **Body:** `multipart/form-data` containing the image file.
- **Response:** A JSON array containing bounding boxes, detected labels, confidence scores, size estimations, and risk categorizations.

---

## 🤝 Contributing
Contributions are welcome! Please create a branch, make your changes, and submit a pull request. Ensure that the tests pass and that `requirements.txt` is updated if you add new dependencies.

## 📄 License
This project is licensed under the MIT License.
