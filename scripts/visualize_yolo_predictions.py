import os
import cv2

# Directory paths
labels_dir = r'C:\Users\Kumaraguru\LPR_Project\dataset\labels\train'
images_dir = r'C:\Users\Kumaraguru\LPR_Project\dataset\images\train'
output_dir = r'C:\Users\Kumaraguru\LPR_Project\visualized_results'
image_extension = '.png'

os.makedirs(output_dir, exist_ok=True)

label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

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

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID:{int(class_id)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    out_path = os.path.join(output_dir, base_name + '_vis.png')
    cv2.imwrite(out_path, img)
    print(f'[✓] Saved visualization: {out_path}')

print('All visualizations complete!')
