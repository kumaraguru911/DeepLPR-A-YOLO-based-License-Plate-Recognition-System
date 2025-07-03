# DeepLPR: A YOLO-based License Plate Recognition System

This project is an end-to-end License Plate Recognition (LPR) system built on YOLOv8 for detection and Tesseract OCR for recognition. It includes scripts for training, inference, evaluation, visualization, and annotation conversion.

---

## Features

- **YOLOv8 License Plate Detection**  
  Detects license plates in images using a custom-trained YOLOv8 model.
- **Tesseract OCR Integration**  
  Extracts license plate text from detected regions.
- **Evaluation Script**  
  Computes precision and recall for predictions against ground truth, with IoU debugging.
- **Visualization Tool**  
  Draws bounding boxes and class labels on images for easy inspection.
- **Annotation Conversion**  
  (Optional) Convert between Pascal VOC XML and YOLO formats.
- **Batch Processing**  
  Supports batch evaluation and visualization for multiple images.

---

## Directory Structure

```
LPR_Project/
├── dataset/
│   ├── images/
│   │   └── train/         # Training/test images
│   └── labels/
│       └── train/         # YOLO-format label files
├── runs/
│   └── detect/
│       └── predict2/      # YOLOv8 prediction results
├── scripts/
│   ├── evaluate_yolo_metrics.py
│   ├── ocr_from_yolo_results.py
│   └── visualize_yolo_predictions.py
├── README.md
└── ...
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/kumaraguru911/DeepLPR-A-YOLO-based-License-Plate-Recognition-System.git
cd DeepLPR-A-YOLO-based-License-Plate-Recognition-System
```

### 2. Install Requirements

- Python 3.8+
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)
- OpenCV
- pytesseract
- numpy

```bash
pip install ultralytics opencv-python pytesseract numpy
```

- **Install Tesseract OCR**:  
  Download from [here](https://github.com/tesseract-ocr/tesseract) and add to your PATH.

### 3. Prepare Data

- Place your images in `dataset/images/train/`
- Place your YOLO-format label files in `dataset/labels/train/`

### 4. Train YOLOv8

```bash
yolo detect train data=your_data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 5. Run Inference

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=dataset/images/train/
```

### 6. Evaluate Predictions

```bash
python scripts/evaluate_yolo_metrics.py
```
- Prints precision, recall, and IoU for each prediction.

### 7. Visualize Predictions

```bash
python scripts/visualize_yolo_predictions.py
```
- Saves images with bounding boxes for easy inspection.

### 8. OCR from Detected Plates

```bash
python scripts/ocr_from_yolo_results.py
```
- Extracts and prints license plate text from detected regions.

---

## Notes

- Make sure the Tesseract executable path is set in `ocr_from_yolo_results.py` if not in your system PATH.
- Adjust paths in scripts as needed for your environment.
- The evaluation script includes IoU debugging output for troubleshooting.

---

## Contributors

- [kumaraguru911](https://github.com/kumaraguru911)

---

## License

This project is licensed under the MIT License.
