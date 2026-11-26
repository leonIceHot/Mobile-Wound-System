from torch.utils.data import DataLoader
#from torch_lr_finder import LRFinder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, \
    precision_recall_fscore_support, average_precision_score
from matplotlib import pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from IPython.display import clear_output
from util.tool import Create_NF, Create_New_File, Create_csv, New_str_loader_img_Y_TF, cls_check_checkpoint, Model_choose
from thop import clever_format, profile
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenetv3
from torch.cuda.amp import autocast, GradScaler
import os
import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchinfo
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import timm
import gc
import csv
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

# 釋放所有未使用的記憶體
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("GPU memory cleared.")

# 由於ResNet模型在訓練時會有隨機變數影響每次訓練的結果
# 使用這段副程式指定一個固定的隨機變數，確保模型在相同條件下進行訓練、檢測
def set_random_seed(seed):
    torch.manual_seed(seed)  # 在CPU中設置固定生成隨機數的種子
    torch.cuda.manual_seed(seed)  # 在GPU中設置固定生成隨機數的種子
    torch.cuda.manual_seed_all(seed)  # 在'所有GPU'中設置固定生成隨機數的種子
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        # alpha 是類別權重，請確保它是一個 Tensor 並在正確的 device 上
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. 計算 Cross Entropy (不加 reduction)
        # inputs 形狀: [Batch, Classes], targets 形狀: [Batch] (整數標籤)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 2. 計算 pt (預測正確類別的機率)
        pt = torch.exp(-ce_loss)

        # 3. 計算 Focal Loss: (1-pt)^gamma * CE
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    """#### Checkpoint setting"""

# def checkpoint_set(checkpoint_point, epoch, best_loss, scheduler = None, train_loss = None, train_acc = None, val_loss = None, val_acc = None, save_dir = results_dir):
#     # checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')

#     if checkpoint_point :
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#         if scheduler and 'scheduler_state_dict' in checkpoint:
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#         epoch = checkpoint['epoch']
#         best_loss = checkpoint['best_loss']

#         #   load loss anf accuracy metrics
#         train_loss = checkpoint.get('train_loss', [])
#         train_acc = checkpoint.get('train_acc', [])
#         val_loss = checkpoint.get('val_loss', [])
#         val_acc = checkpoint.get('val_accu', [])

#         checkpoint_point = False
#         print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")
#         return checkpoint_point, epoch, best_val_accuracy, train_loss, train_acc, val_loss, val_acc

#     else:
#         checkpoint = {
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epoch': epoch,
#             'best_val_accuracy': best_val_accuracy,
#             'train_loss': train_loss,
#             'train_acc': train_acc,
#             'val_loss': val_loss,
#             'val_acc': val_acc
#         }
#         if scheduler and hasattr(scheduler, 'state_dict'):
#             checkpoint['scheduler_state_dict'] = scheduler.state_dict()

#         torch.save(checkpoint, f'{save_dir}/checkpoint_path')
#         print("checkpoint saved.")

"""### Training function"""

'''
def train_1_epoch(model, optimizer, data_loader, device, epoch, epoch_total, criterion):
    total, correct, epoch_train_loss, train_acc = 0, 0, 0.0, 0.0
    warmup_epochs = 5  # 設定前 5 輪為 warmup

    train_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    model.train()       # 將模型轉為訓練模式

    for i, (images, labels) in train_loop:
        train_inputs = images.to(device)
        train_labels = labels.to(device)

        # --- Normal ---
        optimizer.zero_grad()
        train_outputs = model(train_inputs)
        loss = criterion(train_outputs, train_labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(train_outputs.data, 1)
        total += train_labels.size(0)
        correct += (predicted == train_labels).sum().item()

        # 更新進度條顯示
        current_acc = 100 * correct / total
        train_loop.set_description(f'    Epoch [{epoch+1:0>3d}/{epoch_total:0>3d}]')
        train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        epoch_train_loss += loss.item()

    # 計算平均 Loss 與 Accuracy
    epoch_avg_loss = epoch_train_loss / len(data_loader)
    train_acc = 100 * correct / total

    # 注意：train_losses 和 train_accuracies 如果是全域變數，
    # 建議在主迴圈（呼叫處）進行 append，或者確保這兩個 list 在此作用域可見。
    train_losses.append(epoch_avg_loss)
    train_accuracies.append(train_acc)

    return epoch_avg_loss, train_acc
'''


def train_1_epoch(model, optimizer, criterion, data_loader, device, epoch, epoch_total, scaler):
    total, correct, epoch_train_loss = 0, 0, 0.0

    model.train()
    # 使用 tqdm 顯示進度
    train_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

    for i, (images, labels) in train_loop:
        train_inputs = images.to(device)
        train_labels = labels.to(device)

        optimizer.zero_grad()

        # --- 使用混合精度訓練 ---
        # 確保只有在 cuda 時才使用 autocast
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == 'cuda')):
            train_outputs = model(train_inputs)
            loss = criterion(train_outputs, train_labels)

        # 反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- 統計資料 ---
        _, predicted = torch.max(train_outputs.data, 1)
        total += train_labels.size(0)
        correct += (predicted == train_labels).sum().item()
        epoch_train_loss += loss.item()

        # 更新進度條
        current_acc = 100 * correct / total
        train_loop.set_description(f'Epoch [{epoch + 1:0>3d}/{epoch_total:0>3d}]')
        train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

    epoch_avg_loss = epoch_train_loss / len(data_loader)
    train_acc = 100 * correct / total

    return epoch_avg_loss, train_acc


"""### Validation function"""

'''
def val_1_epoch(model, optimizer, data_loader, device, epoch, epoch_total):
    total, correct, epoch_val_loss, val_acc = 0, 0, 0.0, 0.0

    val_loop = tqdm(enumerate(data_loader), total=len(data_loader))
    model.eval()			#將模型改為推論模式

    with torch.no_grad():
        for i, (images, labels) in val_loop:
            val_inputs = images.to(device, non_blocking = True)
            val_labels = labels.to(device, non_blocking = True)

            optimizer.zero_grad()
            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)

            #epoch_val_loss.append(loss)
            #loss = criterion_warmup(val_outputs, val_labels)

            _, predicted = torch.max(val_outputs.data, 1) #選擇信心指數最高的類別
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()	#檢查模型推論是否正確

            val_loop.set_description(f'    Epoch [{epoch+1:0>3d}/{epoch_total:0>3d}]')
            val_loop.set_postfix(crossentropy_Loss=loss.item(), Accuracy=100 * correct / total) #顯示進度條

            epoch_val_loss += loss.item()

    epoch_avg_loss = epoch_val_loss / len(data_loader)
    val_losses.append(epoch_avg_loss)   #將當前epoch訓練的loss記錄到列表(val_losses)中

    val_acc = 100 * correct / total	    #計算準確率
    val_accuracies.append(val_acc)      #將當前epoch訓練的loss記錄到列表(val_accuracies)中

    return epoch_avg_loss, val_acc
'''


def val_1_epoch(model, criterion, data_loader, device, epoch, epoch_total):
    # 初始化統計數值
    total, correct, epoch_val_loss = 0, 0, 0.0

    model.eval()  # 進入評估模式（關閉 Dropout, BatchNorm 使用移動平均）
    val_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

    with torch.no_grad():  # 核心：關閉梯度計算，節省記憶體與運算量
        for i, (images, labels) in val_loop:
            # 將資料搬移至指定設備
            val_inputs = images.to(device, non_blocking=True)
            val_labels = labels.to(device, non_blocking=True)

            # 使用混合精度推論 (與訓練保持一致，可加快推論速度)
            with torch.amp.autocast(device_type="cuda"):
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)

            # 統計正確率與 Loss
            _, predicted = torch.max(val_outputs.data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            epoch_val_loss += loss.item()

            # 更新進度條
            current_acc = 100 * correct / total
            val_loop.set_description(f'    Validation [{epoch + 1:0>3d}/{epoch_total:0>3d}]')
            val_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

    # 計算整輪的平均指標
    epoch_avg_loss = epoch_val_loss / len(data_loader)
    val_losses.append(epoch_avg_loss)  # 將當前epoch訓練的loss記錄到列表(val_losses)中

    val_acc = 100 * correct / total  # 計算準確率
    val_accuracies.append(val_acc)  # 將當前epoch訓練的loss記錄到列表(val_accuracies)中

    # 只回傳結果，不在此處 append 到全域 list，保持函數純粹性
    return epoch_avg_loss, val_acc


"""### Line Chart"""
# 繪製 Loss 曲線
def draw_line_chart():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='cyan', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='gold', linestyle='--', linewidth=2)  # 用虛線區分
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)  # 加入網格，容易看水平線
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(
        os.path.join(results_dir, 'loss_curve.png'))  # os.path.join = 連接前後兩個路徑名 = results_dir路徑\loss_curve.png
    plt.show()

    # 繪製 Accuracy 曲線
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


if __name__ == '__main__':
    """# Parameters setting"""
    clear_memory()

    """ Model Setting"""
    #   model and dataset setting
    model_num = 4
    dataset_num = 0
    worker = 20


    scaler = torch.amp.GradScaler('cuda')  # 建立混合精度縮放器
    #model_list = ['mvitv2_tiny', 'mvitv2_small', 'mvitv2_base']
    model_list = ['mobilenetv5_base','mobilenetv4_small', 'mobilenetv4_medium', 'mobilenetv4_large', 'moblienetV3_large', 'moblienetV3_small']

    # model_list  = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3','efficientnet-b4', 'efficientnet-b5',
    #                'efficientnet-b6', 'efficientnet-b7', 'efficientnetv2-small','efficientnetv2-medium','efficientnetv2-large']
    dataset_list = ['wound1', 'wound2_mix_aug', 'new_wound2+3', 'new_wound4', 'new_wound5', 'GhostFormer']
    dataset_name = dataset_list[dataset_num]
    model_name = model_list[model_num]

    print(f'name = {model_name}')
    print(f'dataset = {dataset_name}')

    #   Parameters setting
    img_size, total_epochs, learning_rate, batch_size = (224, 224), 1000, 2.66E-03, 16

    # 初始化評估指標
    epoch_train_loss = []
    train_losses = []
    epoch_val_loss = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0

    """### Run training and validation"""
    # 訓練參數初始化
    early_stopping = 1000
    early_stopping_patience = 500  # 👉 加入耐心值設定 (請依需求自行調整數值，例如 10 輪沒進步就停止)
    best_loss, train_loss, val_loss = 100, 0.0, 0.0
    train_accuracy, val_accuracy = 0.0, 0.0
    best_acc = 0.0

    # 設置並建立儲存的資料夾
    path = './results/' + 'Mobile' + '/' + model_name
    date = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
    path = path + '/' + dataset_name + '_' + str(learning_rate) + '/' + date

    print(f"path = {path}")
    os.makedirs(path, exist_ok=True)

    weights_dir = path + '/weights'
    results_dir = path + '/results'  # results_dir的值為要建立的資料夾名稱
    checkpoint_dir = path + '/checkpoints'

    print(f'weight direction: {weights_dir}')
    os.makedirs(weights_dir)

    print(f'result direction: {results_dir}')
    os.makedirs(results_dir, exist_ok=True)

    print(f'checkpoint direction: {checkpoint_dir}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # training_log_file = CSV位置, training_log_df = CSV檔案中抬頭分類
    training_log_file, training_log_df = Create_csv(results_dir, "training_log.csv", used='train_result')
    mi_log_file, mi_log_df = Create_csv(results_dir, "mi_log.csv", used='final_result')  # model information

    # 是否使用cuda(gpu)，否則使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.version.cuda)
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("CUDA 版本:", torch.version.cuda)
    print("GPU 數量:", torch.cuda.device_count())
    print("當前 GPU:", torch.cuda.current_device())
    print("GPU 名稱:", torch.cuda.get_device_name(torch.cuda.current_device()))


    # Preprocessing
    set_random_seed(42)  # 執行副程式set_random_seed()



    # 驗證集/測試集保持原樣，不需要修改
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Data Loader
    # 使用ImageFolder('A',transform = 'B') 指定'A'資料夾使用'B'的預處理方式
    # dataset_name = dataset_list[dataset_num]
    # dataset_path = 'D:\\Liz\\models\\dataset\\' + dataset_name
    dataset_path = r'./68_dataset_8p1p5'
    print(f'Dataset: {dataset_path}')
    print(f'Train: {dataset_path}/train')
    print(f'Val: {dataset_path}/val')
    print(f'Test: {dataset_path}/test')

    train_dataset = ImageFolder(dataset_path + '/train', transform=val_test_transform)
    val_dataset = ImageFolder(dataset_path + '/val', transform=val_test_transform)
    test_dataset = ImageFolder(dataset_path + '/test', transform=val_test_transform)

    # 指定批次大小 & 是否打亂資料順序(shuffle)
    # 建構可迭代的資料裝載器(決定資料從哪裡讀取以及如何讀取, batch size, 每個epoch是否亂序)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=worker)

    # 查看train_dataset資料夾有哪些分類(W0、W1、W2、W3...) 之後會用到
    class_names = train_dataset.classes
    print(f'Class = {len(class_names)}, {class_names}')
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    """# Model"""
    num_classes = len(class_names)  # 類別數量

    # EfficientNet
    # if model_num == 1:  # EfficientNet_b1
    #     model = timm.create_model('efficientnet_b1', num_classes=num_classes)
    # elif model_num == 2:  # EfficientNet_b2
    #     model = timm.create_model('efficientnet_b2', num_classes=num_classes)
    # elif model_num == 3:  # EfficientNet_b3
    #     model = timm.create_model('efficientnet_b3', num_classes=num_classes)
    # elif model_num == 4:  # EfficientNet_b4
    #     model = timm.create_model('efficientnet_b4', num_classes=num_classes)
    # elif model_num == 5:  # EfficientNet_b5
    #     model = timm.create_model('efficientnet_b5', num_classes=num_classes)
    # elif model_num == 6:  # EfficientNet_b6
    #     model = timm.create_model('efficientnet_b6', num_classes=num_classes)
    # elif model_num == 7:  # EfficientNet_b7
    #     model = timm.create_model('efficientnet_b7', num_classes=num_classes)
    # elif model_num == 8:  # Efficientv2_small
    #     model = timm.create_model('efficientnetv2_s', num_classes=num_classes)
    # elif model_num == 9:  # Efficientv2_medium
    #     model = timm.create_model('efficientnetv2_m', num_classes=num_classes)
    # elif model_num == 10:  # Efficientv2_large
    #     model = timm.create_model('efficientnetv2_l', num_classes=num_classes)


    # MobileNet
    if model_num == 0:
        model = timm.create_model('mobilenetv5_base', num_classes=num_classes)
    elif model_num == 1:  # mobilenetv4_small
        model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', num_classes=num_classes)
    elif model_num == 2:  # mobilenetv4_medium
        model = timm.create_model('mobilenetv4_conv_medium.e500_r224_in1k', num_classes=num_classes)
    elif model_num == 3:  # mobilenetv4_large
        model = timm.create_model('mobilenetv4_conv_large.e500_r256_in1k', num_classes=num_classes)
    elif model_num == 4:  # MoblieNetV3_large
        model, backbone = Model_choose(class_names, 4)
    elif model_num == 5:  # MoblieNetV3_small
        model, backbone = Model_choose(class_names, 3)

    model = model.to(device)


    # MViTv2
    '''
    if model_num == 0:  # MViTv2_tiny
        model = timm.create_model('mvitv2_tiny', pretrained=True, num_classes=num_classes, img_size=img_size)
    elif model_num == 1:  # MViTv2_small
        model = timm.create_model('mvitv2_small', pretrained=True, num_classes=num_classes, img_size=img_size)
    elif model_num == 2:  # MViTv2_base
        model = timm.create_model('mvitv2_base', pretrained=True, num_classes=num_classes, img_size=img_size)
     
    # --- 核心修正：動態替換分類頭 ---
    # 獲取模型最後一層的輸入維度 (timm 提供的標準接口)
    n_inputs = model.num_features

    # 根據模型結構，將 Dropout 和線性層塞回去
    # MobileViT 系列通常使用 .head，而 MViT 也使用 .head
    # 為了保險，我們可以用 hasattr 檢查，或直接針對你的這幾個模型替換
    if hasattr(model, 'head'):
        model.head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(n_inputs, num_classes)
        )
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(n_inputs, num_classes)
        )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)

    #   模型架構
    # for name, param in model.named_parameters():
    #     print(name)

    #   確認哪些層被解凍
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    """### Architecture"""

    # 模型架構圖
    print(f'{model_name} Architecture')
    # summary(model, input_size=(1, 3, img_size[0], img_size[1]))  # (batch_size, channels, height, width)
    model.to(device)  # 將模型加載到指定設備上

    """### FLOPs and Parameters"""
    tensor = torch.rand(1, 3, img_size[0], img_size[1]).to(device)
    flops, params = profile(model, inputs=(tensor,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f'model = {model_list[model_num]}')
    print("PARAMS: ", params)
    print("FLOPs: ", flops)

    """### Hyperparameters"""
    class_counts = {}
    for class_name in os.listdir(dataset_path + '/train'):
        class_path = os.path.join(dataset_path + '/train', class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            class_counts[class_name] = count

    training_class_counts = [class_counts.get(cls, 0) for cls in class_names]
    print(f'each class counts of training dataset : {training_class_counts}')

    """### 加入 class weights 調整 loss 函數
    #### W𝑖 = N / n𝑖
    > W𝑖：類別 𝑖 的權重
    > N：所有訓練樣本總數
    > n𝑖：類別 𝑖 的樣本數
    """
    class_weights = []
    for c in training_class_counts:
        class_weights.append(round(len(train_dataset) / int(c), 1))
    print(f'class_weights: {class_weights}')
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # 確保轉成 Tensor
    '''

    '''   
    # 設定模型的超參數
    step = len(train_loader)
    warmup_epochs = 5

    # 修改你的損失函數定義
    label_smoothing_value = 0.1
    
    # 1. Warmup 階段
    criterion_warmup = nn.CrossEntropyLoss(label_smoothing=label_smoothing_value)

    # 2. 正式訓練階段 (帶權重 + 平滑)
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing_value)
    
    # 建議從頭到尾只用一個 Loss    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1
    )
    '''

    # MobileNetv3
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999),eps=1e-8,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)


    # MobileNetV4
    '''
    if model_list == 'mobilenetv4':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.05)

        # 1. 定義預熱階段 (從 1e-6 上升到 learning_rate)
        warmup_sche = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )

        # 2. 定義餘弦退火階段 (你截圖中的部分)
        cosine_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )

        # 3. 真正將兩者組合 (這才是完整的 Warmup + Cosine)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sche, cosine_sche],
            milestones=[warmup_epochs]
        )
    '''

    # MViT
    '''
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    '''

    # EfficientNet
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9,momentum=0.9,eps=1e-3,weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)


    model, optimizer, scheduler, epoch_continue, best_loss, best_acc, weight_dir, all_loss, all_acc, train_time, val_time = cls_check_checkpoint(
        model, optimizer, scheduler, weights_dir, checkpoint_dir, dataset_list[dataset_num])
    print(f"used time: {sum(train_time) + sum(val_time)}s")
    print(f"weight saved as : {weight_dir}...")

    for epoch in range(total_epochs - epoch_continue):
        # ---訓練階段---#
        print(
            f"--------------------------Previous Epoch: Epoch {epoch + 1 :03d}/{total_epochs:03d}--------------------------")
        t1 = time.time()
        train_loss, train_accuracy = train_1_epoch(model, optimizer, criterion, train_loader, device, epoch, total_epochs,scaler)
        t2 = time.time()
        # ---訓練階段---#

        # ---驗證階段---#
        t3 = time.time()
        val_loss, val_accuracy = val_1_epoch(model, criterion, val_loader, device, epoch, total_epochs)
        t4 = time.time()

        # 更新學習率
        scheduler.step()  # 更新學習率 (拔除 val_loss)

        clear_output()  # 清除輸出顯示

        # 修正 print total_epochs
        print(
            f"-------------------------- Current Epoch: Epoch {epoch + 1 :03d}/{total_epochs:03d} --------------------------")
        print(f"    Learning rate: {optimizer.param_groups[0]['lr']:.10f}")
        print(f"    Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}%")
        print(f"    Training epoch time: {t2 - t1:.3f} s")
        print(f"    Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}%")
        print(f"    Validation epoch time: {t4 - t3:.3f} s")
        print(f"    Best loss: {best_loss:.10f}")
        print("-------------------------------------------------------------------------------------------------------")

        # Log the results into the DataFrame
        new_row = pd.DataFrame([{
            'Epoch': epoch + 1,
            'Training Loss': train_loss,
            'Training Accuracy': train_accuracy,
            'Training time': t2 - t1,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy,
            'Validation time': t4 - t3,
            'Learning Rate': optimizer.param_groups[0]["lr"]
        }])
        training_log_df = pd.concat([training_log_df, new_row], ignore_index=True)

        # Save log to CSV after each epoch
        training_log_df.to_csv(training_log_file, index=False)
        print(f"Training log saved to {training_log_file}")

        # ========================== 存檔與 Early Stopping 判斷邏輯 ==========================
        model_saved = False  # 👉 新增一個旗標，用來記錄這個 epoch 是否有儲存任何模型

        # ------------------ 1. 判斷 Best Accuracy (越高越好) ------------------
        if val_accuracy > best_acc:
            print(f'🔥 New Best Acc: {val_accuracy:.4f} > Previous Best: {best_acc:.4f}')
            best_acc = val_accuracy  # 更新最高紀錄

            # 存檔路徑 (標記為 best_acc)
            save_path_acc = f'{weights_dir}/epoch{epoch + 1:04d}_ACC_{val_accuracy:.4f}_Loss_{val_loss:.4f}_best_acc.pth'
            torch.save(model.state_dict(), save_path_acc)
            print(f'✅ Model saved (Best Acc) to {save_path_acc}')
            model_saved = True  # 👉 標記已存檔

        # ------------------ 2. 判斷 Best Loss (越低越好) ------------------
        if val_loss < best_loss:
            print(f'💧 New Best Loss: {val_loss:.4f} < Previous Best: {best_loss:.4f}')
            best_loss = val_loss  # 更新最低紀錄

            # 存檔路徑 (標記為 best_loss)
            save_path_loss = f'{weights_dir}/epoch{epoch + 1:04d}_ACC_{val_accuracy:.4f}_Loss_{val_loss:.4f}_best_loss.pth'
            torch.save(model.state_dict(), save_path_loss)
            print(f'✅ Model saved (Best Loss) to {save_path_loss}')
            model_saved = True  # 👉 標記已存檔

        # ------------------ 3. 結算 Early Stopping 計數器 ------------------
        if model_saved:
            early_stopping = 0  # 👉 只要有儲存模型 (破任一紀錄)，Early stopping 計數器就歸零
        else:
            early_stopping += 1  # 👉 如果都沒有破紀錄，計數器 +1
            print(f'⚠️ No improvement in validation metrics for {early_stopping} epoch(s).')

        # ====================================================================================

        if epoch == total_epochs - 1:
            print('save model...')
            torch.save(model.state_dict(),
                       f'{weights_dir}/epoch{epoch + 1:04d}_ACC_{val_accuracy:.4f}_Loss_{val_loss:.4f}_last.pth')
            print('save model done')
            print("Training complete.")

        print(
            f"[  Epoch : {epoch + 1:03d} - Train Loss : {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}% - Validation Loss : {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}%  ]\n")

        torch.save({
            'epoch': epoch + 1 + epoch_continue,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler != None else scheduler,
            'best_loss_score': best_loss,
            'best_acc_score': best_acc,
            'weight_path': weights_dir,
            'matrics_loss': all_loss,
            'matrics_acc': all_acc,
            'train_time': train_time,
            'val_time': val_time,
        }, f"{checkpoint_dir}model_checkpoint_{dataset_name}.tar")

        # ------------------ 4. 觸發 Early Stopping 的條件 ------------------
        if early_stopping >= early_stopping_patience:
            print(
                f"🛑 Early stopping triggered at epoch {epoch + 1}. No improvement for {early_stopping_patience} epochs.")
            torch.save(model.state_dict(),
                       f'{weights_dir}/epoch{epoch + 1:04d}_ACC_{val_accuracy:.4f}_Loss_{val_loss:.4f}_earlystop_last.pth')
            break