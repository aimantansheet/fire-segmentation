# 🔥 Fire Segmentation using U-Net

An end-to-end computer vision project for detecting and segmenting fire regions in images using a U-Net semantic segmentation model with a ResNet34 encoder.

The model was developed using PyTorch and deployed as a REST API using FastAPI, Docker, and Hugging Face Spaces.

---

## 📌 Project Overview

Fire detection is an important computer vision problem for applications such as surveillance, industrial safety, and early warning systems.

Instead of simply classifying an image as "Fire" or "No Fire", this project performs **semantic segmentation** to identify the specific pixels corresponding to fire regions.

### Key Features

- U-Net semantic segmentation architecture
- ResNet34 encoder
- ImageNet-pretrained encoder
- PyTorch-based model training and inference
- Albumentations for image preprocessing and augmentation
- Intersection over Union (IoU) for evaluation
- FastAPI REST API
- Docker containerization
- Hugging Face Spaces deployment
- Postman API testing

---

## 🧠 Model Architecture

The project uses a **U-Net** architecture with a **ResNet34 encoder**.

During training, the ResNet34 encoder was initialized with ImageNet-pretrained weights.

### Architecture

    Input Image
         │
         ▼
    ResNet34 Encoder
         │
         ▼
    U-Net Decoder
         │
         ▼
    1-Channel Output
         │
         ▼
    Sigmoid Activation
         │
         ▼
    Binary Fire Mask

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | U-Net |
| Encoder | ResNet34 |
| Encoder Weights | ImageNet |
| Input Channels | 3 |
| Output Classes | 1 |
| Input Resolution | 256 × 256 |
| Framework | PyTorch |

---

## 📊 Dataset

The project uses the **Fire Segmentation Image Dataset** downloaded from Kaggle.

The dataset contains:

- Fire images
- Non-fire images
- Pixel-level segmentation masks for fire images

Fire images are paired with their corresponding segmentation masks.

For non-fire images, an all-zero segmentation mask is generated because no fire region is present.

The dataset was divided into:

- **80% training**
- **20% validation**

using `train_test_split` with a fixed `random_state=42`.

---

## 🔄 Data Preprocessing & Augmentation

All images and masks were resized to:

    256 × 256

### Training Augmentation

The following augmentations were applied to the training data:

- Horizontal Flip
- Vertical Flip
- Random Brightness/Contrast
- Rotation up to ±20°
- Normalization

### Validation Preprocessing

Validation images were:

- Resized to 256 × 256
- Normalized
- Converted to PyTorch tensors

Albumentations was used to apply transformations consistently to the images and corresponding segmentation masks.

---

## 🏋️ Model Training

The model was trained for **10 epochs**.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | U-Net |
| Encoder | ResNet34 |
| Encoder Initialization | ImageNet |
| Optimizer | Adam |
| Learning Rate | 1 × 10⁻⁴ |
| Batch Size | 32 |
| Epochs | 10 |
| Loss Function | Dice Loss + BCE Loss |
| Evaluation Metric | IoU |
| Image Size | 256 × 256 |

### Loss Function

A combined segmentation loss was used:

    Total Loss = Dice Loss + Binary Cross-Entropy Loss

Dice Loss helps optimize the overlap between predicted and ground-truth segmentation regions, while Binary Cross-Entropy provides pixel-level classification supervision.

---

## 📈 Results

The model showed consistent improvement throughout training.

| Epoch | Train Loss | Validation Loss | Train IoU | Validation IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.7524 | 0.3739 | 0.3957 | 0.5488 |
| 2 | 0.3503 | 0.2884 | 0.5471 | 0.6036 |
| 3 | 0.2907 | 0.2526 | 0.5984 | 0.6413 |
| 4 | 0.2683 | 0.2283 | 0.6208 | 0.6679 |
| 5 | 0.2499 | 0.2216 | 0.6413 | 0.6756 |
| 6 | 0.2421 | 0.2089 | 0.6500 | 0.6907 |
| 7 | 0.2301 | 0.2025 | 0.6641 | 0.6987 |
| 8 | 0.2258 | 0.2014 | 0.6692 | 0.6997 |
| 9 | 0.2168 | 0.1973 | 0.6801 | 0.7046 |
| 10 | **0.2123** | **0.1901** | **0.6857** | **0.7137** |

### Final Validation Performance

**Validation IoU: 0.7137**

The validation IoU improved from **0.5488 in epoch 1 to 0.7137 in epoch 10**.

The validation loss decreased from **0.3739 to 0.1901** over the 10 training epochs.

---

## 🚀 REST API

The trained model was deployed as a REST API using **FastAPI**.

### API Endpoints

#### Health Check

    GET /

Response:

    {
      "message": "Fire Segmentation API is running"
    }

#### Fire Segmentation

    POST /predict

The endpoint accepts an image file and returns the predicted binary segmentation mask as a PNG image.

### API Pipeline

    Image Upload
         ↓
    FastAPI
         ↓
    Image Preprocessing
         ↓
    U-Net Model
         ↓
    Sigmoid Activation
         ↓
    Threshold > 0.5
         ↓
    Binary Segmentation Mask
         ↓
    PNG Response

---

## 🧪 API Testing

The API was tested using **Postman** by sending an image file to the `/predict` endpoint.

### Example Request

    POST /predict
    Content-Type: multipart/form-data

Form-data:

    file → image file

The API returns the predicted segmentation mask in PNG format.

---

## 🐳 Docker

The API is containerized using Docker.

The Docker image:

1. Uses Python 3.11
2. Installs the required Python dependencies
3. Copies the application files
4. Exposes port 7860
5. Runs the FastAPI application using Uvicorn

### Build the Docker Image

    docker build -t fire-segmentation-api .

### Run the Container

    docker run -p 7860:7860 fire-segmentation-api

The API can then be accessed at:

    http://localhost:7860

FastAPI interactive documentation:

    http://localhost:7860/docs

---

## ☁️ Deployment

The API has been deployed using **Hugging Face Spaces** with Docker.

### Live Deployment

🔥 **[Fire Segmentation API on Hugging Face](https://huggingface.co/spaces/aiman72366/Fire_Segmentation_API)**

The deployed application provides an API endpoint for performing fire segmentation on uploaded images.

---

## ⚙️ Model Weights

The trained model checkpoint is:

    best_unet_fire.pth

The checkpoint is approximately **98 MB**.

Due to its size, the trained model weights are **not stored in this GitHub repository**.

The trained checkpoint is maintained separately with the deployed Hugging Face Space.

### Model Weight Availability

The trained model checkpoint is available as part of the deployed Hugging Face application.

👉 **[Access the Hugging Face Space](https://huggingface.co/spaces/aiman72366/Fire_Segmentation_API)**

The inference code in `model.py` expects the checkpoint to be available as:

    best_unet_fire.pth

when running the API locally.

> **Note:** The GitHub repository contains the source code, training notebook, API implementation, Docker configuration, and documentation. The trained model checkpoint is maintained separately on Hugging Face.

---

## 📁 Project Structure

    fire-segmentation/
    │
    ├── app.py
    ├── model.py
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    ├── .gitignore
    │
    └── notebooks/
        └── FireSegmentation.ipynb

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Segmentation Models PyTorch**
- **OpenCV**
- **NumPy**
- **Albumentations**
- **FastAPI**
- **Uvicorn**
- **Docker**
- **Hugging Face Spaces**
- **Postman**
- **Kaggle Dataset**

---

## 🎯 Key Learning Outcomes

This project demonstrates an end-to-end computer vision and machine learning workflow, including:

- Image dataset preparation
- Semantic segmentation
- U-Net model development
- Transfer learning using ResNet34
- Image preprocessing and augmentation
- Combined Dice + BCE loss
- Model evaluation using IoU
- PyTorch model inference
- REST API development with FastAPI
- Docker containerization
- Cloud deployment
- API testing with Postman

---

## 🔮 Future Improvements

Potential improvements to the project include:

- Training for more epochs
- Hyperparameter tuning
- Experimenting with different segmentation architectures
- Evaluating additional metrics such as Dice Score and F1 Score
- Adding fire-mask overlays to the API response
- Adding confidence/probability visualization
- Implementing automated model downloading for easier local setup
- Adding automated testing and CI/CD

---

## 👤 Author

**Aiman Tansheet**

MS Data Science | Data Science & AI/ML

GitHub: [aimantansheet](https://github.com/aimantansheet)
