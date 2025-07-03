import os
import cv2
import pytesseract

# 🟢 If you installed Tesseract but it's not in PATH, set the path manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# Path to YOLO's detection images and labels (train set)
labels_dir = r'C:\Users\Kumaraguru\LPR_Project\dataset\labels\train'
images_dir = r'C:\Users\Kumaraguru\LPR_Project\dataset\images\train'
image_extension = '.png'

if not os.path.exists(labels_dir):
    print(f"[!] Labels directory not found: {labels_dir}")
    exit(1)

label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
if not label_files:
    print(f"[!] No label files found in {labels_dir}")
    exit(1)

for label_file in label_files:
    base_name = os.path.splitext(label_file)[0]
    image_file = os.path.join(images_dir, base_name + image_extension)
    if not os.path.exists(image_file):
        print(f"[!] No image found for label {label_file}, skipping.")
        continue

    img = cv2.imread(image_file)
    if img is None:
        print(f"[!] Failed to load image: {image_file}, skipping.")
        continue

    img_h, img_w = img.shape[:2]

    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)

        # Convert YOLO format back to pixel coords
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        x1 = max(0, int(x_center - width / 2))
        y1 = max(0, int(y_center - height / 2))
        x2 = min(img_w, int(x_center + width / 2))
        y2 = min(img_h, int(y_center + height / 2))

        # Skip invalid crops
        if x1 >= x2 or y1 >= y2:
            print(f"[!] Invalid crop for plate {i+1} in {image_file}, skipping.")
            continue

        cropped_plate = img[y1:y2, x1:x2]
        if cropped_plate.size == 0:
            print(f"[!] Empty crop for plate {i+1} in {image_file}, skipping.")
            continue

        # Optional: apply preprocessing for better OCR
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run Tesseract OCR
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        print(f"[✓] {image_file} Plate {i+1} text: {text.strip()}")

print("OCR complete!")
# Note: Ensure Tesseract is installed and configured correctly.
# You can adjust the Tesseract config options for better results based on your dataset.