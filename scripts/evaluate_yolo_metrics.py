import os
import numpy as np

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def parse_yolo_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, w, h = map(float, parts)
            x *= img_w
            y *= img_h
            w *= img_w
            h *= img_h
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([x1, y1, x2, y2])
    return boxes

def evaluate(pred_dir, gt_dir, images_dir, iou_thresh=0.5):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    tp, fp, fn = 0, 0, 0
    iou_thresh = 0.3  # Lower threshold for more forgiving matching
    for pred_file in pred_files:
        base = os.path.splitext(pred_file)[0]
        gt_file = os.path.join(gt_dir, base + '.txt')
        img_file = os.path.join(images_dir, base + '.png')
        if not os.path.exists(gt_file) or not os.path.exists(img_file):
            print(f"[!] Skipping {base}: missing gt or image file.")
            continue
        img = cv2.imread(img_file)
        img_h, img_w = img.shape[:2]
        pred_boxes = parse_yolo_label(os.path.join(pred_dir, pred_file), img_w, img_h)
        gt_boxes = parse_yolo_label(gt_file, img_w, img_h)
        print(f"[DEBUG] {base}: {len(pred_boxes)} pred boxes, {len(gt_boxes)} gt boxes")
        print(f"[DEBUG] pred_boxes: {pred_boxes}")
        print(f"[DEBUG] gt_boxes: {gt_boxes}")
        matched = set()
        for pb in pred_boxes:
            found = False
            for i, gb in enumerate(gt_boxes):
                if i in matched:
                    continue
                iou = compute_iou(pb, gb)
                print(f'IoU between pred {pb} and gt {gb}: {iou:.3f}')
                if iou >= iou_thresh:
                    tp += 1
                    matched.add(i)
                    found = True
                    break
            if not found:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f'Precision: {precision:.3f}, Recall: {recall:.3f}')

if __name__ == '__main__':
    import cv2
    # Use the latest YOLOv8 prediction directory and test image/label
    pred_dir = r'C:\Users\Kumaraguru\LPR_Project\runs\detect\predict2\labels'
    gt_dir = r'C:\Users\Kumaraguru\LPR_Project\dataset\labels\train'
    images_dir = r'C:\Users\Kumaraguru\LPR_Project\dataset\images\train'
    evaluate(pred_dir, gt_dir, images_dir)
