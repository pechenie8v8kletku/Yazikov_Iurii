import os
import json
import random
import torch
import numpy as np
import dlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from utils import seed_everything
from Dataset import Point_face
from models import (
    FirstModel,
    SecondModel,
    ThirdModel,
    FourthModel,
    FifthModel,
    SixthModel,
    SeventhModel
)
# гиперпараметры
BATCH_SIZE = 64
NUM_WORKERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 67
# метаданные после мультипроцессинга
META_300W_TEST = "meta_test_300w.json"
META_MEPRO_TEST = "meta_test_mepro.json"

# имя модели + путь до лучшего сейва
MODEL_PATHS = {
    "ResNet18_MLP_MSE": "models/Resnet18_MLP_MSE/best_model.pt",
    "ResNet18_smoothL1_avgpool": "models/Resnet18_smoothL1_avgpool/best_model.pt",
    "ResNet18_smoothL1_nopool": "models/Resnet18_smoothL1_nopool/best_model.pt",
    "ResNet18_GraphLaplacian": "models/Resnet18_GraphLaplacian_avgpool/best_model.pt",
    "EfficientNet_B2_MSE": "models/Efficent_net_b2_MLP_MSE/epoch_99.pt",
    "EfficientNet_B2_GraphLaplacian": "models/EfficientNet_B2_GraphLaplacian_avgpool/best_model.pt",
    "EfficientNet_B3_GraphLaplacian_dif_alpha": "models/EfficientNet_B3_GraphLaplacian+smoothL1_trying_fit_alpha_avgpool/best_model.pt",
    "EfficientNet_B3_GraphLaplacian_smoothL1": "models/EfficientNet_B3_GraphLaplacian+smoothL1_avgpool/best_model.pt",
    "MobileNetV3_MSE": "models/MobileNETV3_MLP_MSE/epoch_83.pt",
    "MobileViT_S_MSE": "models/Mobilevit_s_MSE_/best_model.pt",
    "ConvNeXt_Tiny_MSE": "models/ConvNeXt_tiny_MSE_avgpool/best_model.pt",
    "ConvNeXt_Tiny_GraphLaplacian": "models/ConvNeXt_tiny_GraphLaplacian_avgpool/best_model.pt",
}
# путь до установленной модели для библиотеки dlib
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
seed_everything(SEED)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_trained_model(model_class, weights_path):
    model = model_class()
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    return model.to(DEVICE)

# получаем предсказания модели для удобного дальнейшего анализа
@torch.no_grad()
def get_model_predictions(model, loader, raw_meta):
    model.eval()
    predictions = {}
    meta_keys = list(raw_meta.keys())
    global_idx = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        bboxes = batch["orig_bbox"].numpy()
        shapes = batch["crop_shape"].numpy()

        outputs = model(images).cpu().numpy()
        for i in range(images.size(0)):
            x1, y1, x2, y2 = bboxes[i]
            crop_w, crop_h = shapes[i]
            pred_norm = outputs[i].reshape(68, 2)
            pred_orig = pred_norm.copy()
            pred_orig[:, 0] = pred_orig[:, 0] * crop_w + x1
            pred_orig[:, 1] = pred_orig[:, 1] * crop_h + y1

            key = meta_keys[global_idx]
            predictions[key] = pred_orig
            global_idx += 1

    return predictions

# предсказания модели dlib
def get_dlib_predictions(raw_meta):
    print("Запуск детектора особых точек DLIB...")
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    predictions = {}

    for key, meta in raw_meta.items():
        x1, y1, x2, y2 = meta["bbox"]
        rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))

        img_orig = Image.open(meta["image_path"]).convert('RGB')
        img_np = np.array(img_orig)
        shape = predictor(img_np, rect)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)

        predictions[key] = pts
    return predictions
# считаем ошибку
def compute_ced_errors(predicted_points, raw_meta):
    errors = []
    for key, pred_pts in predicted_points.items():
        gt_pts = np.array(raw_meta[key]["landmarks"], dtype=np.float32)
        x1, y1, x2, y2 = raw_meta[key]["bbox"]
        norm_factor = np.sqrt((y2 - y1) * (x2 - x1))
        distances = np.linalg.norm(gt_pts - pred_pts, axis=1)
        img_error = np.mean(distances) / (norm_factor + 1e-6)
        errors.append(img_error)

    return np.sort(errors)

# считаем auc с учетом того что отсекаем на 0.08
def compute_ced_auc(sorted_errors, max_thr=0.08, step=0.0005):
    proportions = np.arange(sorted_errors.shape[0], dtype=np.float32) / sorted_errors.shape[0]
    auc = 0.0
    for thr in np.arange(0.0, max_thr, step):
        gt_indexes = np.flatnonzero(sorted_errors >= thr)
        first_gt_idx = gt_indexes[0] if len(gt_indexes) > 0 else len(sorted_errors) - 1
        auc += proportions[first_gt_idx] * step
    return auc / max_thr

# создает полотно с 16 изображениями
def save_qualitative_canvas(all_methods_preds, raw_meta):
    print("Генерация полотен визуализации...")
    all_keys = list(raw_meta.keys())
    random_keys = random.sample(all_keys, min(16, len(all_keys)))

    for method_name, preds_dict in all_methods_preds.items():
        fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=120)
        axes = axes.flatten()

        for idx, key in enumerate(random_keys):
            meta = raw_meta[key]
            x1, y1, x2, y2 = meta["bbox"]
            img = Image.open(meta["image_path"]).convert('RGB')
            img_crop = img.crop((max(0, x1 - 10), max(0, y1 - 10), x2 + 10, y2 + 10))
            shift_x, shift_y = max(0, x1 - 10), max(0, y1 - 10)

            ax = axes[idx]
            ax.imshow(img_crop)
            gt = np.array(meta["landmarks"], dtype=np.float32)
            ax.scatter(gt[:, 0] - shift_x, gt[:, 1] - shift_y, s=12,
                       label="Ground Truth" if idx == 0 else "")
            if key in preds_dict:
                pts = preds_dict[key]
                ax.scatter(pts[:, 0] - shift_x, pts[:, 1] - shift_y,  s=10,
                           label=f"Pred: {method_name}" if idx == 0 else "")

            ax.set_title(f"ID: {os.path.basename(meta['image_path'])}", fontsize=10)
            ax.axis("off")
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=18)
        plt.tight_layout()

        output_path = f"visual_canvas_mepro_{method_name}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


# проверка на датасете dlib Обучен на библиотеке 300w но тем не менее он все равно будет на графике чисто для наглядности
def evaluate_dataset(meta_path, dataset_name, include_dlib=False):
    print(f"\n================ Оцениваем датасет: {dataset_name} ================")
    with open(meta_path, 'r') as f:
        raw_meta = json.load(f)

    test_dataset = Point_face(meta_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,persistent_workers=True,
                             pin_memory=True)

    all_dataset_preds = {}

    model_classes = {
        "ResNet18_MLP_MSE": FirstModel,
        "ResNet18_smoothL1_avgpool": FirstModel,
        "ResNet18_GraphLaplacian": FirstModel,
        "ResNet18_smoothL1_nopool": FourthModel,
        "EfficientNet_B2_MSE": SecondModel,
        "EfficientNet_B2_GraphLaplacian": SecondModel,
        "EfficientNet_B3_GraphLaplacian_smoothL1": SeventhModel,
        "EfficientNet_B3_GraphLaplacian_dif_alpha": SeventhModel,
        "MobileNetV3_MSE": ThirdModel,
        "MobileViT_S_MSE": SixthModel,
        "ConvNeXt_Tiny_MSE": FifthModel,
        "ConvNeXt_Tiny_GraphLaplacian": FifthModel,
    }

    for model_name, model_class in model_classes.items():
        model = load_trained_model(model_class, MODEL_PATHS[model_name])
        preds = get_model_predictions(model, test_loader,raw_meta)
        all_dataset_preds[model_name] = preds


    dlib_preds = get_dlib_predictions(raw_meta)
    all_dataset_preds["DLIB"] = dlib_preds


    plt.figure(figsize=(10, 7), dpi=100)

    for method_name, preds_dict in all_dataset_preds.items():
        errors = compute_ced_errors(preds_dict, raw_meta)
        auc_score = compute_ced_auc(errors, max_thr=0.08)
        print(f">> Метрика AUC для {method_name} на {dataset_name}: {auc_score:.4f}")

        proportion = np.arange(errors.shape[0], dtype=np.float32) / errors.shape[0]
        visible_idx = np.flatnonzero(errors <= 0.08)

        plt.plot(
            errors[visible_idx],
            proportion[visible_idx],
            label=f"{method_name} (AUC: {auc_score:.3f})",
            linewidth=2
        )

    plt.xlim(0.0, 0.08)
    plt.ylim(0.0, 1.0)
    plt.title(f"Cumulative Error Distribution (CED) - {dataset_name}", fontsize=14)
    plt.xlabel("Normalized Error (Threshold 0.08)", fontsize=12)
    plt.ylabel("Proportion of Images", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)

    chart_path = f"ced_plot_{dataset_name}.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {chart_path}")

    return all_dataset_preds, raw_meta


def main():

    _ = evaluate_dataset(META_300W_TEST, "300W", include_dlib=False)
    mepro_preds, mepro_meta = evaluate_dataset(META_MEPRO_TEST, "Menpo", include_dlib=True)
    save_qualitative_canvas(mepro_preds, mepro_meta)


if __name__ == "__main__":
    main()