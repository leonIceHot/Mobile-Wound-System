# ----- 深度學習核心 (PyTorch) -----
import torch                                      # PyTorch 核心
import torch.nn as nn                             # 神經網路模組 (損失函數、層定義等)
import torch.optim as optim                       # 優化器 (SGD, Adam 等)
import torchvision.models as models               # 預訓練模型 (ResNet, EfficientNet 等)
import torchvision.transforms as transforms       # 影像前處理與資料增強
from torchvision.datasets import ImageFolder      # 自動依資料夾名稱載入分類資料集
from torch.utils.data import DataLoader           # 批次資料載入器

# ----- 模型視覺化 / 可解釋性 (XAI) -----
from pytorch_grad_cam import GradCAM                          # Grad-CAM 熱力圖 (可視化模型關注區域)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # 指定 Grad-CAM 的目標類別
from pytorch_grad_cam.utils.image import show_cam_on_image    # 將 Grad-CAM 熱力圖疊加在原圖上
from captum.attr import LayerGradCam                          # Captum 的 Layer Grad-CAM (另一套 XAI 工具)

# ----- 評估指標 (Scikit-learn) -----
from sklearn.metrics import (
    confusion_matrix,                  # 混淆矩陣
    classification_report,             # 精確率/召回率/F1 報告
    ConfusionMatrixDisplay,            # 混淆矩陣視覺化
    precision_recall_fscore_support,   # 各類別的 Precision, Recall, F1, Support
    average_precision_score            # AP (Average Precision) 分數
)

# ----- 數值運算 / 影像處理 -----
import timm
import numpy as np                     # 數值陣列運算
import cv2                             # OpenCV 影像處理 (讀圖、縮放、色彩轉換等)
from PIL import Image                  # Pillow 影像讀取與格式轉換

# ----- 繪圖 -----
import matplotlib
matplotlib.use('Agg')                  # 非互動式後端 (適用於伺服器 / 無 GUI 環境，存圖用)
import matplotlib.pyplot as plt        # 繪製圖表 (曲線圖、混淆矩陣等)

# ----- 檔案與路徑處理 -----
import os                              # 作業系統介面 (建立資料夾、路徑操作)
import shutil                          # 高階檔案操作 (複製、移動、刪除資料夾)
from pathlib import Path               # 物件導向式路徑處理 (比 os.path 更直觀)
import glob                            # 萬用字元檔案搜尋 (e.g., *.jpg)

# ----- 進度條 -----
from tqdm import tqdm                  # 在迴圈中顯示進度條

# ----- 資料處理工具 -----
import pandas as pd                    # 表格資料處理 (DataFrame 讀寫 CSV/Excel)
import json                            # JSON 格式讀寫 (設定檔、結果儲存)
import csv                             # CSV 格式讀寫
import re                              # 正規表達式 (字串搜尋與比對)
import time                            # 計時 (訓練時間統計)

# ----- 警告抑制 -----
import warnings
warnings.filterwarnings('ignore')      # 忽略所有警告訊息 (保持輸出整潔)

# 全局變量定義 (移動到文件頂部)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet 均值
IMAGENET_STD = [0.229, 0.224, 0.225]  # ImageNet 標準差

def set_random_seed(seed):
    torch.manual_seed(seed)  # ?CPU?????????????
    torch.cuda.manual_seed(seed)  # ?GPU?????????????
    torch.cuda.manual_seed_all(seed)  # ?'??GPU'?????????????
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize_image(image):
    """?????????? [0, 1] ?????"""
    image = image * np.array(IMAGENET_STD).reshape(1, 1, -1) + np.array(IMAGENET_MEAN).reshape(1, 1, -1)  # ????
    image = np.clip(image, 0, 1)  # ????? [0, 1] ?
    return image

def draw_line_chart(train_accuracies, val_accuracies, train_losses, val_losses, results_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='cyan', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='gold', linestyle='--', linewidth=2)  # ?????
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)  # ???????????
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(
        os.path.join(results_dir, 'loss_curve.png'))  # os.path.join = ????????? = results_dir??\loss_curve.png
    plt.show()

    # ?? Accuracy ??
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy', color='cyan', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='gold', linestyle='--', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'))
    plt.show()

def show_error_result(image_arr, label_name, pred_name, num_rows, num_cols, image_num):
    fig = plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for i in range(len(image_arr)):
        trans_images = np.transpose(image_arr[i], (1, 2, 0))  # (3, 244, 244) -> (244, 244, 3)
        trans_images = denormalize_image(trans_images)  # ????
        err_label = label_name[i]
        err_pred = pred_name[i]

        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.imshow(trans_images)  # ??????????
        plt.title(f"No.{image_num[i]}\nLabel: {err_label}\nPred: {err_pred}", fontsize=8)
        plt.axis('off')  # ?????
    return fig  # 返回 fig 对象以便保存

def cal_error_result(labels, predictions, images, class_names, num_rows, num_cols, test_dataset):
    err_arr = []
    label_name = []
    pred_name = []
    error_num = []  # ?????????

    for i in range(len(test_dataset)):
        label = int(labels[i])
        pred = int(predictions[i])
        if class_names[label] != class_names[pred]:
            err_arr.append(images[i])
            label_name.append(class_names[label])
            pred_name.append(class_names[pred])
            error_num.append(i)

    print("num_errors = ", len(err_arr))
    # show_error_result 現在返回 fig 對象
    fig = show_error_result(err_arr, label_name, pred_name, num_rows, num_cols, error_num)

    return fig

#   show images and the correctness of predictions in a table
def display_predictions(images, labels, predictions, class_names, test_dataset, test_correct, results_dir, num_cols=5):
    err = len(test_dataset) - test_correct
    photo_num = err
    if photo_num == 0:
        print("No errors to display.")
        return

    #   ???????
    num_rows = (photo_num + num_cols - 1) // num_cols  # ???????

    #   ?????????????
    # cal_error_result 現在返回 fig 對象
    fig = cal_error_result(labels, predictions, images, class_names, num_rows, num_cols, test_dataset)

    fig.savefig(os.path.join(results_dir, f'display_predictions.png'))
    print(f'Total in testing dataset = {len(test_dataset)}, correct = {test_correct}, error = {err}')

    plt.tight_layout()  # ???????????
    plt.show()
    plt.close(fig)  # 關閉圖形以釋放內存

def vit_reshape_transform(tensor):
    # ????????????????????? CLS Token
    # size 36864 / 768 = 48?
    # ?? size ? 36864?????? Channel ?

    b, n, c = tensor.shape
    # ?????????? 6x8 ? 7x7
    # ????????????
    h = int(np.sqrt(n))
    w = n // h

    result = tensor.reshape(b, h, w, c)
    result = result.permute(0, 3, 1, 2)
    return result

class GradCAM:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 確保模型在評估模式
        self.model.eval()

        # 清空梯度
        self.model.zero_grad()

        # 傳播
        input_tensor.requires_grad_(True)  # 確保輸入張量需要梯度
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # 計算目標類別的損失
        # 這裡的 loss 必須是標量，因此取 output[0, class_idx]
        if output.dim() == 2:  # 檢查 output 是否是 (batch_size, num_classes)
            loss = output[0, class_idx]
        else:  # 如果 output 是 (num_classes,)
            loss = output[class_idx]

        # 反向傳播
        loss.backward(retain_graph=True if input_tensor.grad_fn else False)  # 如果之後會用到計算圖，retain_graph=True

        gradients = self.gradients
        activations = self.activations

        # 確保梯度和激活不是 None
        if gradients is None or activations is None:
            raise RuntimeError("Gradients or Activations are None. Check your target layer and hooks.")

        # 應用 reshape_transform
        if self.reshape_transform:
            gradients = self.reshape_transform(gradients)
            activations = self.reshape_transform(activations)

        # 確保 gradients 和 activations 是 CPU numpy 數組
        gradients = gradients.cpu().data.numpy()[0]  # 取出 batch 中的第一個樣本
        activations = activations.cpu().data.numpy()[0]  # 取出 batch 中的第一個樣本

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        # 確保 resize 的目標尺寸是整數
        target_size = (int(input_tensor.shape[3]), int(input_tensor.shape[2]))
        cam = cv2.resize(cam, target_size)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

def save_cam_result(img_tensor, mask, title, save_path):
    img = img_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())  # ?? 0-1

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # ??
    cam_bgr = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, cam_bgr)

def save_comparison_plot(img_tensor, mask, pred_idx, pred_conf, gt_idx, class_names, save_path, img_name=None):
    """
    保存比較 Grad-CAM 的圖片

    Args:
        img_tensor: 圖片張量 (Tensor) - 形狀: (C, H, W) 或 (1, C, H, W)
        mask: CAM 遮罩 (NumPy Array) - 形狀: (H, W)
        pred_idx: 預測類別索引
        pred_conf: 預測信心值
        gt_idx: 真實類別索引
        class_names: 類別名稱列表
        save_path: 儲存路徑
        img_name: 原始檔案名稱（例如 "WCC_0_015.jpg"）
    """
    try:
        # 處理維度
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        img = img_tensor.cpu().detach().numpy()
        if img.shape[0] in [1, 3, 4]:
            img = img.transpose(1, 2, 0)

        if img.shape[2] == 1:
            img = img.squeeze(2)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        # 調整 mask 尺寸
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # 產生熱力圖
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        # 尺寸對齊
        if img.shape != heatmap.shape:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        cam_overlay = heatmap * 0.4 + img * 0.6
        cam_overlay = np.clip(cam_overlay, 0, 1)

        # 繪圖
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        is_correct = (pred_idx == gt_idx)
        status_text = "✔ CORRECT" if is_correct else "✘ ERROR"
        title_color = "green" if is_correct else "red"

        # 左圖：原始影像
        axes[0].imshow(img)
        axes[0].set_title(
            f"Original Image\nGT: {class_names[gt_idx]} \n img_name: {img_name}",
            fontsize=11,
            fontweight='bold'
        )
        axes[0].axis('off')

        # 右圖：Grad-CAM
        axes[1].imshow(cam_overlay)
        pred_text = f"Pred: {class_names[pred_idx]}\n Confidence: {pred_conf:.2%}"
        axes[1].set_title(
            f"Grad-CAM [{status_text}]\n{pred_text}\n img_name: {img_name}",
            fontsize=11,
            fontweight='bold',
            color=title_color
        )
        axes[1].axis('off')

        # 背景色
        fig.patch.set_facecolor('white' if is_correct else '#ffe6e6')

        # 主標題：加入檔案名稱
        if img_name:
            suptitle_text = (
                f"File: {img_name}\n"
                f"Prediction: {class_names[pred_idx]} | Ground Truth: {class_names[gt_idx]}"
            )
        else:
            suptitle_text = (
                f"Prediction: {class_names[pred_idx]} | Ground Truth: {class_names[gt_idx]}"
            )

        plt.suptitle(
            suptitle_text,
            fontsize=12,
            fontweight='bold'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    except Exception as e:
        print(f"❌ 儲存圖片失敗 ({save_path}): {e}")
        plt.close('all')
        raise


def run_full_gradcam_analysis(model, test_loader, class_names, device, results_dir,
                              base_dir="gradcam_analysis", max_images=100):
    """
    執行完整的 Grad-CAM 分析並保存結果

    Args:
        model: 訓練好的模型
        test_loader: 測試資料加載器（shuffle=False）
        class_names: 類別名稱列表
        device: 計算設備 (cpu 或 cuda)
        results_dir: 結果保存的根目錄
        base_dir: Grad-CAM 分析結果的子目錄名稱
        max_images: 最多分析的圖片數量
    """

    correct_dir = os.path.join(results_dir, base_dir, "correct")
    error_dir = os.path.join(results_dir, base_dir, "error")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    print(f"Results will be saved in: {os.path.join(results_dir, base_dir)}")

    model.eval()

    # 取得模型名稱並選擇目標層
    print(model._get_name())
    name = model._get_name()

    if name == 'MultiScaleVit':
        last_stage = model.stages[-1]
        last_block = last_stage.blocks[-1]
        target_layer = last_block.norm2
    else:
        target_layer = model.features[-1]

    print(f"Using Captum LayerGradCam with target layer: {target_layer.__class__.__name__}")
    layer_gc = LayerGradCam(model, target_layer)

    # ★ 取得底層 dataset 的 samples
    dataset_samples = test_loader.dataset.samples  # [(path, label), ...]

    count = 0
    global_idx = 0  # 用來追蹤在 dataset 中的絕對索引

    print(f"Generating Grad-CAMs... (up to {max_images} images)")

    for images, labels in tqdm(test_loader, desc="Generating Grad-CAMs"):
        #if count >= max_images:
        #    break

        images_gpu = images.to(device)
        labels_gpu = labels.to(device)

        with torch.no_grad():
            outputs = model(images_gpu)
            preds = torch.argmax(outputs, dim=1)
            confidences = torch.softmax(outputs, dim=1).max(dim=1).values

        for i in range(images_gpu.size(0)):
            if count >= max_images:
                break

            pred_idx = preds[i].item()
            gt_idx = labels_gpu[i].item()
            confidence = confidences[i].item()
            img_single = images_gpu[i:i + 1]

            # ★ 取得原始檔案路徑和檔名
            img_path, _ = dataset_samples[global_idx]
            img_name = os.path.basename(img_path)

            # 計算 Grad-CAM
            attributions = layer_gc.attribute(img_single, target=pred_idx)

            # 處理 CAM
            grayscale_cam = attributions.squeeze().cpu().detach().numpy()

            if grayscale_cam.ndim == 3:
                grayscale_cam = grayscale_cam.mean(axis=0)

            grayscale_cam = np.abs(grayscale_cam)
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

            # 決定保存路徑（正確或錯誤）
            if pred_idx == gt_idx:
                save_path = os.path.join(
                    correct_dir,
                    f"{count:03d}_correct_{class_names[gt_idx]}_conf{confidence:.2f}.png"
                )
            else:
                save_path = os.path.join(
                    error_dir,
                    f"{count:03d}_error_GT_{class_names[gt_idx]}_PRED_{class_names[pred_idx]}_conf{confidence:.2f}.png"
                )

            # ★ 保存圖片，傳入檔名
            save_comparison_plot(
                img_single,
                grayscale_cam,
                pred_idx,
                confidence,
                gt_idx,
                class_names,
                save_path,
                img_name=img_name  # ★ 傳入原始檔名
            )

            count += 1
            global_idx += 1  # ★ 遞增全域索引

            if count % 10 == 0:
                print(f"  Processed: {count}/{max_images}")

    print(f"\n✅ Grad-CAM analysis completed! Saved to '{os.path.join(results_dir, base_dir)}'")
    print(f"   - Correct predictions: {len(os.listdir(correct_dir))} images")
    print(f"   - Error predictions: {len(os.listdir(error_dir))} images")




def re_evaluate_two_class(logits, class_pair):
    # logits: ??????? 1D tensor?shape: num_classes?
    # class_pair: tensor([clsA, clsB]) ?? tensor([1, 3])

    clsA, clsB = class_pair[0], class_pair[1]
    if logits[clsA] > logits[clsB]:
        return clsA.item()
    else:
        return clsB.item()

def plot_confusion_matrix(model, dataloader, classes, dataset_name, device, results_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []  # 儲存所有的輸出 logits

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())  # 儲存輸出

    # 轉換為 numpy array
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 生成混淆矩陣
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    print(f"{dataset_name} Confusion Matrix:\n{cm}\n")

    # 繪製混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{dataset_name} Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{dataset_name.lower()}_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 計算 precision, recall, F1-score，添加 zero_division 參數避免警告
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        labels=range(len(classes)),
        zero_division=0  # ★ 關鍵：避免警告
    )

    ap = []

    # 計算 softmax 概率（正確的做法）
    exp_outputs = np.exp(all_outputs - np.max(all_outputs, axis=1, keepdims=True))  # 數值穩定性
    softmax_scores = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

    for i, classname in enumerate(classes):
        # 檢查此類別是否在標籤中出現過
        if i in all_labels:
            # 創建二元標籤（1 表示該類別，0 表示其他）
            binary_actuals = (all_labels == i).astype(int)

            # 使用該類別的 softmax 概率作為分數
            scores = softmax_scores[:, i]

            # 檢查是否有正樣本和負樣本
            if np.unique(binary_actuals).size > 1:  # 同時有 0 和 1
                try:
                    current_ap = average_precision_score(binary_actuals, scores)
                    ap.append(current_ap)
                except Exception as e:
                    print(f"Warning: Could not calculate AP for class {classname}: {e}")
                    ap.append(np.nan)
            else:
                # 只有正樣本或只有負樣本
                ap.append(np.nan)
        else:
            # 類別不在測試資料中
            ap.append(np.nan)

    # 打印結果
    print(f'{dataset_name} Metrics:')
    print("-" * 110)
    print(f'{"Class":<20} | {"Precision":>10} | {"Recall":>10} | {"F1-Score":>10} | {"Support":>8} | {"AP":>10}')
    print("-" * 110)

    for i, classname in enumerate(classes):
        ap_str = f"{ap[i]:.3f}" if not np.isnan(ap[i]) else "N/A"
        print(
            f'{classname:<20} | {precision[i]:>10.3f} | {recall[i]:>10.3f} | {f1[i]:>10.3f} | {support[i]:>8d} | {ap_str:>10s}')

    print("-" * 110)

    # 計算並打印平均指標
    valid_ap = [x for x in ap if not np.isnan(x)]
    avg_ap = np.mean(valid_ap) if valid_ap else 0.0

    print(
        f'{"Average":<20} | {np.mean(precision):>10.3f} | {np.mean(recall):>10.3f} | {np.mean(f1):>10.3f} | {np.sum(support):>8d} | {avg_ap:>10.3f}')
    print("-" * 110)
    print()

def bar_draw(x_axis, y_axis, dataset_name, results_dir):
    plt.figure(figsize=(15, 5))
    plt.bar(x_axis, y_axis)
    plt.xlabel('class')
    plt.ylabel('Class prediction')
    plt.title('Number of Correct Predictions per Class')
    plt.savefig(os.path.join(results_dir,
                             dataset_name + '_pred_bar.png'))  # os.path.join = ????????? = results_dir??\loss_curve.png
    plt.show()
    plt.close()

def calculate(classes, all_preds, all_labels):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_ap = 0
    num_classes_with_data = 0  # 追踪實際有數據的類別數量

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None,
                                                               labels=range(len(classes)))
    ap = []

    for i, classname in enumerate(classes):
        binary_actuals = [1 if label == i else 0 for label in all_labels]
        binary_preds_for_ap = [1 if pred == i else 0 for pred in all_preds]

        if np.unique(binary_actuals).size > 1:  # 確保有正樣本和負樣本
            current_ap = average_precision_score(binary_actuals, binary_preds_for_ap)
            ap.append(current_ap)
            total_precision += precision[i]
            total_recall += recall[i]
            total_f1 += f1[i]
            total_ap += current_ap
            num_classes_with_data += 1
        else:
            ap.append(np.nan)  # 該類別只有單一標籤，AP 不適用

    if num_classes_with_data > 0:
        print(
            f'Average Precision = {total_precision / num_classes_with_data:.3f}, Average Recall = {total_recall / num_classes_with_data:.3f}, Average F1-score = {total_f1 / num_classes_with_data:.3f}, Average AP = {total_ap / num_classes_with_data:.3f}\n')
    else:
        print("No classes with sufficient data to calculate average metrics.")

def cls_check_checkpoint(model, optimizer, scheduler, weights_dir, checkpoint_dir, dataset_name):
    # 這個函數應該根據你的訓練邏輯來實現
    # 這裡只提供一個簡化的版本，用於加載最佳模型權重
    print(f"Attempting to load model from: {weights_dir}")
    best_acc = 0.0
    best_loss = float('inf')
    epoch_continue = 0
    train_time = []
    val_time = []
    all_loss = []
    all_acc = []
    weight_dir = ""  # 佔位符，如果你的 checkpoint_dir 結構不同，可能需要調整

    # 1. 查找 weights_dir 中所有的 best_acc.pth 文件
    search_pattern = os.path.join(weights_dir, '*best_acc.pth')
    best_acc_files = glob.glob(search_pattern)

    if len(best_acc_files) > 0:
        # 找到 ACC 最高的權重文件
        def extract_acc(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'ACC_([0-9]+\.[0-9]+)', filename)
            if match:
                return float(match.group(1))
            return 0.0

        best_checkpoint_path = max(best_acc_files, key=extract_acc)
        best_acc_val = extract_acc(best_checkpoint_path)

        print(f"Loading best model checkpoint from: {best_checkpoint_path} with ACC: {best_acc_val}")
        checkpoint = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))

        # 清理檢查點，移除不需要的鍵（例如來自 thop 的計數）
        cleaned_checkpoint = {k: v for k, v in checkpoint.items()
                              if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(cleaned_checkpoint)

        best_acc = best_acc_val
        # 如果你的檢查點保存了其他信息，可以在這裡加載
        # 例如:
        # epoch_continue = checkpoint.get('epoch', 0) + 1
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    else:
        print("No best_acc checkpoint found. Starting training from scratch or using pre-trained weights only.")

    # 返回這些值，即使它們是初始值
    return model, optimizer, scheduler, epoch_continue, best_loss, best_acc, weight_dir, all_loss, all_acc, train_time, val_time

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
            model.load_state_dict(cleaned_checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        print("Pretrained model loaded successfully!")
        return model
    else:
        print(f"Pretrained model path '{pretrained_path}' not found!")
        print("Using default ImageNet pretrained weights from timm.")
        return model

def main():
    # Model and dataset Setting
    model_num = 1
    dataset_num = 0
    scaler = torch.cuda.amp.GradScaler()

    #model_list = ['mvitv2_tiny', 'mvitv2_small', 'mvitv2_base']
    model_list = ['mobilenet_v3_small','mobilenetv3_large']

    print("模型選擇")
    model_name = model_list[model_num]
    print(f'name = {model_name}')
    worker = 20

    # -------------------------------------------------------------------------------
    # Parameters setting
    img_size, total_epochs, learning_rate, batch_size = (224, 224), 1000, 2.66e-03, 16

    run_path = './'+model_name
    weights_dir = os.path.join(run_path, 'weights')
    results_dir = os.path.join(run_path, 'results')

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Run Path: {run_path}")
    print(f'Weight direction: {weights_dir}')
    print(f'Result direction: {results_dir}')

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

    # Preprocessing
    set_random_seed(42)

    train_transform = transforms.Compose([
        # 1. 放寬裁切範圍，增加大小與長寬比的變化 (這裡以 50%~100% 為例)
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),

        # 2. 隨機水平翻轉 (保留)
        transforms.RandomHorizontalFlip(p=0.5),

        # 3. 新增：隨機旋轉 (例如正負 15 度)
        transforms.RandomRotation(degrees=15),

        # 4. 新增：隨機改變亮度、對比、飽和度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        # 轉成 Tensor
        transforms.ToTensor(),

        # 正規化 (保留)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # 5. 新增：隨機遮擋 (注意：RandomErasing 必須放在 ToTensor 之後)
        # p=0.5 表示有 50% 的機率觸發遮擋
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    # 驗證集/測試集保持原樣，不需要修改
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data Loader
    dataset_path = r'./68_dataset_8p1p5'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        print("Please ensure the dataset is correctly placed or adjust the path.")
        return

    print(f'Dataset: {dataset_path}')
    print(f'Train: {dataset_path}/train')
    print(f'Val: {dataset_path}/val')
    print(f'Test: {dataset_path}/test')

    if not os.path.exists(os.path.join(dataset_path, 'train')):
        print(f"Error: Training data directory '{dataset_path}/train' not found.")
        return
    if not os.path.exists(os.path.join(dataset_path, 'val')):
        print(f"Error: Validation data directory '{dataset_path}/val' not found.")
        return
    if not os.path.exists(os.path.join(dataset_path, 'test')):
        print(f"Error: Test data directory '{dataset_path}/test' not found.")
        return

    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, 'val'), transform=val_test_transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=worker)

    class_names = train_dataset.classes
    print(f'Class = {len(class_names)}, {class_names}')
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # -------------------------------------------------------------------------------
    #   模型初始化
    num_classes = len(class_names)


    # 選項1: 使用 ImageNet 預訓練權重（推薦）
    print("\n--- Loading Pretrained Model ---")
    '''
    model = None
    if model_num == 0:  # MViTv2_tiny
        model = timm.create_model('mvitv2_tiny', pretrained=True, num_classes=num_classes, img_size=img_size[0])
    elif model_num == 1:  # MViTv2_small
        model = timm.create_model('mvitv2_small', pretrained=True, num_classes=num_classes, img_size=img_size[0])
    elif model_num == 2:  # MViTv2_base
        model = timm.create_model('mvitv2_base', pretrained=True, num_classes=num_classes, img_size=img_size[0])
    else:
        raise ValueError("Invalid model_num. Choose 0, 1, or 2.")
    '''

    if model_num == 0:
        model = models.mobilenet_v3_small(pretrained=True)
    elif model_num == 1:
        model = models.mobilenet_v3_large(pretrained=True)
    
    # 獲取最後一層的輸入特徵數
    in_features = model.classifier[-1].in_features

    # 替換最後的分類層
    model.classifier[-1] = nn.Linear(in_features, num_classes)


    # 選項2: 如果要載入自己訓練的預訓練模型
    # 取消註解以下行並指定正確的路徑
    pretrained_weights_path =  r'./results_5/20260731_19-58-00/weights/epoch0522_ACC_71.0983_Loss_1.1126_best_acc.pth'
    model = load_pretrained_model(model, pretrained_weights_path, device)
    model.to(device)

    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999),eps=1e-8,weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)

    # 1. ????????? weights_dir ??????? best_acc ???
    # 在執行測試之前，假設 weights_dir 已經包含了訓練好的模型
    # 如果你沒有訓練，你需要提供一個有效的權重路徑
    # 這裡會嘗試從 weights_dir 中加載權重
    print("\n--- Model Loading ---")

    """## Testing function"""
    # 假設 `training_log.csv` 已經由訓練過程生成
    # 為了演示，如果文件不存在，我們創建一個假的
    training_log_path = os.path.join(results_dir, 'training_log.csv')
    if not os.path.exists(training_log_path):
        print(f"Warning: '{training_log_path}' not found. Generating dummy data for plotting.")
        dummy_data = {
            'Epoch': list(range(1, 11)),
            'Training Loss': [2.0 - i * 0.1 for i in range(10)],
            'Validation Loss': [2.1 - i * 0.12 for i in range(10)],
            'Training Accuracy': [50.0 + i * 4 for i in range(10)],
            'Validation Accuracy': [48.0 + i * 4.5 for i in range(10)]
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy.to_csv(training_log_path, index=False)
        print("Dummy 'training_log.csv' created.")

    # 載入 training_log.csv
    df = pd.read_csv(training_log_path)
    train_accuracy_series = df['Training Accuracy']
    val_accuracy_series = df['Validation Accuracy']
    train_loss_series = df['Training Loss']
    val_loss_series = df['Validation Loss']

    # 繪製訓練/驗證曲線
    draw_line_chart(train_accuracy_series, val_accuracy_series, train_loss_series, val_loss_series, results_dir)

    # 3. Model Testing (????????)
    model.eval()
    test_correct, total = 0, 0
    all_batch_times = []
    all_test_images = []
    all_test_labels = []
    all_test_predictions = []

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
        'Training Accuracy': train_accuracy_series.iloc[-1] if not train_accuracy_series.empty else np.nan,
        'Validation Accuracy': val_accuracy_series.iloc[-1] if not val_accuracy_series.empty else np.nan,
        'Testing Accuracy': testing_accuracy
    }

    # ????????
    # Extract scalar values from results
    keys = list(results_summary.keys())
    values = [float(v) for v in results_summary.values()]

    plt.figure(figsize=(8, 6))
    plt.bar(keys, values, color=['blue', 'orange', 'green'])
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy (%)')
    plt.title('Dataset Accuracy')
    plt.ylim(0, 100)  # 設置 Y 軸範圍
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center', va='bottom')  # 顯示數值
    plt.savefig(os.path.join(results_dir, 'dataset_accuracy.png'))
    plt.show()
    plt.close()

    print("\n--- Plotting Confusion Matrices ---")
    plot_confusion_matrix(model, train_loader, class_names, 'Training', device, results_dir)
    plot_confusion_matrix(model, val_loader, class_names, 'Validation', device, results_dir)
    plot_confusion_matrix(model, test_loader, class_names, 'Testing', device, results_dir)

    """#### ??per-class precision, recall, F1-score, and average precision"""
    y_pred_all = []
    y_true_all = []
    model.eval()

    print("\n--- Calculating Classification Report ---")
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(test_loader, desc="Calculating Metrics")):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            _, preds = torch.max(output, 1)  # preds?????
            y_pred_all.extend(preds.view(-1).detach().cpu().numpy())
            y_true_all.extend(target.view(-1).detach().cpu().numpy())

    classification_reports = classification_report(y_true=y_true_all,
                                                   y_pred=y_pred_all,
                                                   target_names=class_names,  # 使用 class_names 顯示每個類別
                                                   output_dict=True,
                                                   zero_division=0  # 處理沒有預測到的類別
                                                   )
    print("\nClassification Report:")
    print(classification_reports)

    # 將分類報告保存為 CSV
    DF = pd.DataFrame(classification_reports).transpose()
    DF.to_csv(os.path.join(results_dir, "classification_report.csv"))
    print(f"Classification report saved to {os.path.join(results_dir, 'classification_report.csv')}")

    # 錯誤預測圖片顯示
    print("\n--- Displaying Error Predictions ---")
    # 注意：all_test_images 是一個列表，包含每個測試樣本的 numpy 數組 (C, H, W)
    # display_predictions 需要這些圖片
    display_predictions(np.array(all_test_images), all_test_labels, all_test_predictions, class_names, test_dataset,
                        test_correct, results_dir)

    """#### ????"""
    # ???? (????? 50 ??)
    print("\n--- Running Grad-CAM Analysis ---")
    run_full_gradcam_analysis(model, test_loader, class_names, device, results_dir, max_images=107)

    print("\n--- All tasks completed ---")

if __name__ == '__main__':
    main()
