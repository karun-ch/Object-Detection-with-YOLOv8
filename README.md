# Object-Detection-with-YOLOv8 (Computer Vision) 🎯

## 📌 Project Overview
This project demonstrates **Object Detection** using **YOLOv8** (You Only Look Once) from **Ultralytics**. The model detects objects in images and videos efficiently, making it ideal for applications in **AI, robotics, surveillance, and automation**.

## 🚀 Features
- **Pre-trained YOLOv8 model** (`yolov8n.pt` for fast inference).
- **Single Google Colab notebook** for training and inference.
- **Uses MS COCO dataset** for object detection.
- **Detects multiple objects** (people, cars, animals, etc.).
- **Saves and displays detection results**.

## 📂 Dataset
- The model is pre-trained on the **MS COCO dataset**.
- No need to manually download the dataset; the model is ready to use.

## 🛠️ Installation & Setup
### 1️⃣ Install Required Libraries
Run the following command in **Google Colab**:
```bash
pip install ultralytics opencv-python torch torchvision matplotlib
```

### 2️⃣ Load YOLOv8 Model in Python
```python
import torch
from ultralytics import YOLO

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # YOLOv8 Nano model (lightweight and fast)
```

## 🎯 Object Detection on an Image
```python
# Run inference on an example image
image_path = 'https://ultralytics.com/images/zidane.jpg'  # Sample image URL
results = model(image_path)

# Show detection results
for r in results:
    r.show()
    r.save("output.jpg")  # Save output image
```

## 🔍 Predict Objects in a Custom Image
```python
import cv2
import matplotlib.pyplot as plt

def detect_objects(image_path):
    results = model(image_path)  # Run YOLO model
    for i, r in enumerate(results):
        r.save(f"output_{i}.jpg")  # Save detected image
        display_output(f"output_{i}.jpg")  # Display image

def display_output(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# Example Usage
detect_objects("https://ultralytics.com/images/zidane.jpg")
```

## 🖥️ Running on Google Colab
- Upload the **notebook** to **Google Colab**.
- Ensure **GPU is enabled** for faster processing (`Runtime > Change runtime type > GPU`).

## 🔮 Future Improvements
- **Live Webcam Object Detection**.
- **Custom Training on New Datasets**.
- **Deploy as a Web App using Flask or Streamlit**.

---
🚀 **Get Started Now!** Experiment with different images and tweak parameters to build a powerful **AI-powered object detection system!** 🎯

