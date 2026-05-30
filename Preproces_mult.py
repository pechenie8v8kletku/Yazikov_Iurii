import os
import json
import dlib
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

detector = None


def init_worker():
    global detector
    detector = dlib.get_frontal_face_detector()


def parse_pts_file(pts_path):
    landmarks = []
    with open(pts_path, 'r') as f:
        lines = f.readlines()
    start_reading = False
    for line in lines:
        line = line.strip()
        if line == "{":
            start_reading = True
            continue
        if line == "}":
            break
        if start_reading and line:
            coords = list(map(float, line.split()))
            if len(coords) == 2:
                landmarks.append(coords)

    return landmarks


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def process_single_image(fname, dataset_dir, is_mepro, padding_ratio):
    img_path = os.path.join(dataset_dir, fname)
    base_name = os.path.splitext(fname)[0]
    pts_path = os.path.join(dataset_dir, base_name + '.pts')

    if not os.path.exists(pts_path):
        return None

    landmarks = parse_pts_file(pts_path)

    if is_mepro and len(landmarks) != 68:
        return None

    landmarks_np = np.array(landmarks)
    gt_x1 = int(np.min(landmarks_np[:, 0]))
    gt_y1 = int(np.min(landmarks_np[:, 1]))
    gt_x2 = int(np.max(landmarks_np[:, 0]))
    gt_y2 = int(np.max(landmarks_np[:, 1]))
    gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]

    img = cv2.imread(img_path)
    if img is None:
        return None
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dlib_rects = detector(gray, 1)
    if len(dlib_rects) == 0:
        return None

    best_iou = -1
    best_box = None

    for rect in dlib_rects:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        current_box = [x1, y1, x2, y2]
        iou = calculate_iou(gt_box, current_box)

        if iou > best_iou:
            best_iou = iou
            best_box = current_box

    if best_box is None:
        return None

    if "train" in dataset_dir.lower() and best_iou < 0.3:
        return None

    x1, y1, x2, y2 = best_box
    w_box = x2 - x1
    h_box = y2 - y1

    dw = int(w_box * padding_ratio)
    dh = int(h_box * padding_ratio)

    padded_x1 = max(0, x1 - dw)
    padded_y1 = max(0, y1 - dh)
    padded_x2 = min(w_img, x2 + dw)
    padded_y2 = min(h_img, y2 + dh)

    return base_name, {
        "image_path": img_path,
        "bbox": [padded_x1, padded_y1, padded_x2, padded_y2],
        "landmarks": landmarks
    }


def process_dataset_folder_parallel(dataset_dir, is_mepro=False, padding_ratio=0.15):
    metadata = {}

    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(valid_extensions)]

    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        futures = [
            executor.submit(process_single_image, fname, dataset_dir, is_mepro, padding_ratio)
            for fname in all_files
        ]

        folder_name = os.path.basename(os.path.normpath(dataset_dir))
        parent_name = os.path.basename(os.path.dirname(os.path.normpath(dataset_dir)))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {parent_name}/{folder_name}"):
            res = future.result()
            if res is not None:
                base_name, data = res
                metadata[base_name] = data

    return metadata


if __name__ == "__main__":
    train_300w = process_dataset_folder_parallel("data/300W/train", is_mepro=False)
    test_300w = process_dataset_folder_parallel("data/300W/test", is_mepro=False)
    train_mepro = process_dataset_folder_parallel("data/Menpo/train", is_mepro=True)
    test_mepro = process_dataset_folder_parallel("data/Menpo/test", is_mepro=True)

    with open("meta_train_300w.json", "w") as f:
        json.dump(train_300w, f, indent=4)

    with open("meta_test_300w.json", "w") as f:
        json.dump(test_300w, f, indent=4)

    with open("meta_train_mepro.json", "w") as f:
        json.dump(train_mepro, f, indent=4)

    with open("meta_test_mepro.json", "w") as f:
        json.dump(test_mepro, f, indent=4)

    print("Предобработка завершена")