# ----- 深度學習核心 (PyTorch) -----
import torch                                      # PyTorch 核心
import torch.nn as nn                             # 神經網路模組 (損失函數、層定義等)
import torch.optim as optim                       # 優化器 (SGD, Adam 等)
import torchvision.models as models               # 預訓練模型 (ResNet, EfficientNet 等)
import torchvision.transforms as transforms       # 影像前處理與資料增強
from torchvision.datasets import ImageFolder      # 自動依資料夾名稱載入分類資料集
from torch.utils.data import DataLoader           # 批次資料載入器

# ----- 數值運算 / 影像處理 -----
import timm
import numpy as np                     # 數值陣列運算
import cv2                             # OpenCV 影像處理 (讀圖、縮放、色彩轉換等)
from PIL import Image                  # Pillow 影像讀取與格式轉換
from ultralytics import YOLO, RTDETR

# ----- 檔案與路徑處理 -----
from pathlib import Path               # 物件導向式路徑處理 (比 os.path 更直觀)
import os                              # 作業系統介面 (建立資料夾、路徑操作)
import shutil                          # 高階檔案操作 (複製、移動、刪除資料夾)
import glob                            # 萬用字元檔案搜尋 (e.g., *.jpg)
import sys

# ----- 進度條 -----
from tqdm import tqdm                  # 在迴圈中顯示進度條

# ----- 資料處理工具 -----
import datetime
import pandas as pd                    # 表格資料處理 (DataFrame 讀寫 CSV/Excel)
import json                            # JSON 格式讀寫 (設定檔、結果儲存)
import csv                             # CSV 格式讀寫
import re                              # 正規表達式 (字串搜尋與比對)
import time                            # 計時 (訓練時間統計)


def load_pretrained_model(model, pretrained_path, device):
    """
    載入預訓練模型權重
    """
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)

        # 如果 checkpoint 是完整的字典（包含 state_dict 等），提取 state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        # 如果直接是 state_dict
        elif isinstance(checkpoint, dict):
            # 清理 thop 相關的鍵值
            cleaned_checkpoint = {k: v for k, v in checkpoint.items()
                                  if 'total_ops' not in k and 'total_params' not in k}
            model.load_state_dict(cleaned_checkpoint, strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        print("Pretrained model loaded successfully!")
        return model
    else:
        print(f"Pretrained model path '{pretrained_path}' not found!")
        print("Using default ImageNet pretrained weights from timm.")
        return model


def detection():

    """模型訓練"""
    # 儲存路徑設定
    project = './results/Training'
    date = datetime.date.today()
    name = f'{date} yolov26s_wound'
    output_dir = os.path.join(project, name)
    log_file = os.path.join(output_dir,"Training.log")
    os.makedirs(output_dir,exist_ok=True)


    """測試集運行"""
    # 定義測試結果路徑
    testResult_Dir = f'./results/testing/{name}' #結果儲存路徑
    predictionsResult_Dir = os.path.join(testResult_Dir,'predictions')
    pictureResult_Dir = os.path.join(testResult_Dir,'cropped_picture')
    os.makedirs(predictionsResult_Dir,exist_ok =True)
    os.makedirs(pictureResult_Dir,exist_ok =True)
    csv_dir = os.path.join(testResult_Dir,'csv')
    os.makedirs(csv_dir, exist_ok=True)
    output_file = os.path.join(csv_dir, 'performance_metrics.csv')
    print('資料夾建立完成!')


    print("date =", date)
    print("results exists:", os.path.exists("./results"))
    print("Training exists:", os.path.exists("./results/Training"))
    print("\nTraining subfolders:")

    if os.path.exists("./results/Training"):
        print(os.listdir("./results/Training"))

    print("\nAll best.pt:")
    for p in glob.glob("./results/**/best.pt", recursive=True):
        print(p)

    model = YOLO(f'./bestyolo26/yolo26s/best.pt')
    print('Model Load Successfully! ')
    #print(model.model)

    metrics = model.val(
        data='data/Wound2cutmix.yaml',
        split='test',
        batch=8,
        imgsz=640,
        device=0,
        save=True # save results
    )
    default_save_dir = metrics.save_dir
    metrics_dict = metrics.results_dict

    # 列印性能指標
    print("\n性能指標:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    # 將測試的結果保存
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics_dict.items():
            writer.writerow([key, value])
    print(f"\n性能指标已保存到: {output_file}")

    # 移動到指定位置
    if os.path.exists(default_save_dir):
        for file_name in os.listdir(default_save_dir):
            full_file_name = os.path.join(default_save_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, os.path.join(predictionsResult_Dir, file_name))
        print(f"已將測試結果移動至: {predictionsResult_Dir}")
    else:
        print(f"結果仍然保存於: {default_save_dir}")

    """保存bounding box結果(測試集)"""
    imageDir = "./data/Wound2cutmix/images/test"
    print(imageDir)

    image_files = [f for f in os.listdir(imageDir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("未找到圖片")

    for image_name in image_files:
        image_path = os.path.join(imageDir, image_name)

        results = model(image_path)

        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法載入: {image_path}")
            continue

        height, width, _ = image.shape

        # 5. 裁剪并保存每个目标
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)  # 获取 bounding box 坐标
            class_id = int(results[0].boxes.cls[i])  # 获取类别 ID

            # 获取类别名称（若 `model.names` 不存在，使用默认名称）
            class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"

            # 按类别创建存储文件夹
            class_dir = os.path.join(pictureResult_Dir, f"{class_name}")
            os.makedirs(class_dir, exist_ok=True)

            # 計算放大兩倍後的邊界框
            width = x2 - x1
            height = y2 - y1

            # 中心點
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 新的寬高（兩倍）
            new_width = width * 2
            new_height = height * 2

            # 新的邊界（確保不超出圖片範圍）
            img_h, img_w = image.shape[:2]
            x1_new = max(0, cx - new_width // 2)
            y1_new = max(0, cy - new_height // 2)
            x2_new = min(img_w, cx + new_width // 2)
            y2_new = min(img_h, cy + new_height // 2)

            # 裁剪放大後的區域
            cropped = image[y1_new:y2_new, x1_new:x2_new]
            if cropped.size == 0:
                print(f"無效裁剪區域: {image_name} ({x1_new}, {y1_new}, {x2_new}, {y2_new})")
                continue

            # 生成保存路径
            cropped_filename = f"{os.path.splitext(image_name)[0]}_class{class_id}_{x1}_{y1}.jpg"
            cropped_filepath = os.path.join(class_dir, cropped_filename)

            # 保存裁剪后的图片
            cv2.imwrite(cropped_filepath, cropped)
            print(f"已保存: {cropped_filepath}")


def classification():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    dataset_path = r'./results/testing/2026-08-28 yolov26s_wound/cropped_picture/Wound'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        print("Please ensure the dataset is correctly placed or adjust the path.")
        sys.exit(1)  # <-- ADDED: stop execution if path is missing

    print(f'Dataset: {dataset_path}')

    class_names = ['WCC_0', 'WCC_1', 'WCC_2', 'WCC_3', 'WCC_4', 'other']
    print(f'Class = {len(class_names)}, {class_names}')

    num_classes = len(class_names)
    print("\n--- Loading Pretrained Model ---")
    model = models.mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    pretrained_weights_path = r'/media/hank/Disk/Experiment/Wound/Wound_detection/YOLOv10n/results_8/20260827_17-52-26/weights/epoch0008_ACC_81.7876_Loss_0.8646_best_acc.pth'
    model = load_pretrained_model(model, pretrained_weights_path, device)
    model.to(device)

    output_dir = "./classified_results2"
    print(f'Output directory: {output_dir}')

    for cls_name in class_names:
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    test_images = [f for f in Path(dataset_path).iterdir()
                   if f.suffix.lower() in IMG_EXTS]

    print(f"共找到 {len(test_images)} 張測試圖片")

    # ADDED: fail fast if the detection stage produced 0 crops
    if len(test_images) == 0:
        print(f"錯誤: 在 '{dataset_path}' 中找不到任何圖片。")
        print("這通常表示前面的 YOLO 偵測階段沒有偵測到任何傷口 (0 detections)，")
        print("導致沒有裁切圖片被存到此資料夾。請檢查偵測階段的 conf 閾值或權重檔案。")
        sys.exit(1)

    with torch.no_grad():
        for img_path in test_images:
            image = Image.open(img_path).convert("RGB")
            input_tensor = val_test_transform(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_class = class_names[pred_idx]

            dst_path = os.path.join(output_dir, pred_class, img_path.name)
            shutil.move(img_path, dst_path)
            print(f"{img_path.name} -> {pred_class}")

    print("分類完成！")

    model.eval()
    test_correct, total = 0, 0
    all_batch_times = []
    all_test_images = []
    all_test_labels = []
    all_test_predictions = []

    worker = 20
    test_dataset = ImageFolder(output_dir, transform=val_test_transform, allow_empty=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=worker)
    print(f"Number of test samples: {len(test_dataset)}")

    print("\n--- Running Test Set Inference ---")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            batch_start = time.time()  # ?? batch ????

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            batch_end = time.time()  # ?? batch ????
            all_batch_times.append(batch_end - batch_start)

            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # 收集所有測試結果用於後續分析 (Grad-CAM, 錯誤顯示等)
            all_test_images.append(images.cpu().squeeze(0).numpy())  # 存儲原始圖像 (去除 batch 維度)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    avg_time_per_batch = sum(all_batch_times) / len(all_batch_times)
    total_test_time = sum(all_batch_times)
    mean_fps = total / total_test_time if total_test_time > 0 else 0  # 避免除以零
    testing_accuracy = 100 * test_correct / total
    print(f"\nTesting Accuracy: {testing_accuracy:.2f}%")
    print(f"Mean time per batch: {avg_time_per_batch:.4f}s")
    print(f"Total testing time (inference only): {total_test_time:.4f}s")
    print(f"Mean FPS: {mean_fps:.2f} images/s")

    results_summary = {
        'Testing Accuracy': testing_accuracy
    }


if __name__ == '__main__':

    """### Device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


    detection()
    classification()


