
# Accident Detection System 🚗💥

A deep learning-based accident detection system using CNN and YOLOv8, capable of classifying accident images and drawing bounding boxes around detected elements. Upon detecting an accident, the system sends an alert via a messaging service.

---

## 📌 Features

- **Image Classification** using a custom CNN model.
- **Object Detection** using YOLOv8 (`ultralytics`).
- **Real-time Alerts** sent when an accident is detected.
- Automatically saves annotated images of detected accidents.

---

## 📁 Project Structure

```bash
.
├── accident_ditection_system.py
├── dataa/
│   ├── training/
│   ├── valid/
│   └── testing/
├── RES/             # Stores images where accidents were detected
└── msg.py           # Contains logic to send messages (e.g., SMS or email)
```

---

## 🧠 Models Used

- **CNN** (Custom Sequential model):
  - 3 Convolutional Layers
  - 1 Dense Hidden Layer (512 neurons)
  - Binary output: `Accident` vs `Non-accident`
- **YOLOv8**:
  - Pretrained YOLOv8n model (`yolov8n.pt`) from Ultralytics
  - Used for object localization and bounding boxes

---

## ⚙️ Requirements

Install the required packages via pip:

```bash
pip install tensorflow keras opencv-python numpy matplotlib ultralytics
```

Ensure the following structure for the dataset:
- `dataa/training/` and `dataa/valid/` each should contain:
  - `accident/` and `non_accident/` folders with images

---

## 🚀 Usage

1. Place your training, validation, and testing data in the appropriate folders.
2. Run the script:

```bash
python accident_ditection_system.py
```

3. The script will:
   - Train the CNN on the dataset
   - Use YOLO to detect objects in test images
   - Predict accident presence using the CNN
   - Annotate and save images with "Accident" status
   - Trigger alerts for accident cases

---

## 🛠️ To-Do

- Integrate real-time video stream detection
- Improve alert system with GPS and user info
- Optimize model performance and inference time

---

## 📧 Contact

For queries or contributions, reach out to the project maintainer.
