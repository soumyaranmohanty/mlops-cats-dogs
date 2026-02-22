# 🐶🐱 Cats vs Dogs Classification — End-to-End MLOps Pipeline

## 📌 Project Overview
This project demonstrates a complete end-to-end MLOps pipeline for a binary image classification problem (Cats vs Dogs).

## 🧱 Tech Stack
- PyTorch
- MLflow
- DVC
- FastAPI
- Docker
- GitHub Actions
- Docker Compose
- Pytest

## 📂 Project Structure
mlops-cats-dogs/
├── src/
├── tests/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── model.pt
├── smoke_test.sh
├── deploy.sh
├── evaluate.py

## 🚀 Features
- Data versioning using DVC
- CNN model using PyTorch
- MLflow tracking
- FastAPI inference API
- Docker containerization
- CI/CD with GitHub Actions
- Monitoring and evaluation

## ⚙️ Setup
```bash
pip install -r requirements.txt
python -m src.models.train
uvicorn src.api.app:app --reload
```

## 🧪 Testing
```bash
pytest
```

## 🐳 Docker

### Build and Run Locally
```bash
docker build -t cats-dogs-api .
docker run -p 8000:8000 cats-dogs-api
```

### Run Deployment (Docker Compose)
```bash
docker-compose up -d
```

## 🔄 CI/CD Pipeline

### CI Pipeline
**Trigger:** Push to `main` branch

**Steps:**
1. Install dependencies
2. Run tests (pytest)
3. Build Docker image
4. Push to Docker Hub

The CI pipeline ensures code quality and creates deployable artifacts on every push to the main branch.

### CD Pipeline
**Trigger:** CI success

**Deployment Method:** SSH to remote server/PC

The CD pipeline automatically deploys the application to the production environment once CI checks pass successfully.

### Smoke Testing
After deployment, run smoke tests to verify the service is operational:
```bash
./smoke_test.sh
```

## 📊 Monitoring
The application includes built-in monitoring capabilities:
- **Request Logging:** All API requests are logged
- **Latency Tracking:** Response time monitoring
- **Metrics Endpoint:** `/metrics` endpoint for request count and performance metrics
