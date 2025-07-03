import os
import xml.etree.ElementTree as ET
from PIL import Image

# Paths
images_folder = 'C:/Users/Kumaraguru/LPR_Project/dataset/images/train'
annotations_folder = 'C:/Users/Kumaraguru/LPR_Project/annotations'
labels_folder = 'C:/Users/Kumaraguru/LPR_Project/dataset/labels/train'

# Make sure labels folder exists
os.makedirs(labels_folder, exist_ok=True)

# Iterate over each XML file in annotations
for xml_file in os.listdir(annotations_folder):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(annotations_folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get corresponding image size
    img_filename = root.find('filename').text
    img_path = os.path.join(images_folder, img_filename)
    if not os.path.exists(img_path):
        print(f"[!] Image not found: {img_path}")
        continue
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Prepare YOLO label lines
    yolo_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # For now, we assume single class: license_plate → id 0
        class_id = 0

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Write YOLO TXT
    label_filename = os.path.splitext(xml_file)[0] + '.txt'
    label_path = os.path.join(labels_folder, label_filename)
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    print(f"[✓] Converted {xml_file} → {label_filename}")

print("Conversion complete!")
