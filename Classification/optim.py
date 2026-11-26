import torch.nn as nn
import timm
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import mobilenetv3
from torchvision.datasets import ImageFolder
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader
from util.tool import Model_choose

if __name__ == '__main__':
    model_num = 3
    worker = 4
    img_size = 224
    batch_size = 16

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

    val_test_transform = transforms.Compose([
        # 使用計算好的 resize_size
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = r'./68_dataset_8p1p5'
    print(f'Dataset: {dataset_path}')
    print(f'Train: {dataset_path}/train')
    print(f'Val: {dataset_path}/val')
    print(f'Test: {dataset_path}/test')

    # 1. 載入資料集
    train_dataset = ImageFolder(dataset_path + '/train', transform=val_test_transform)
    val_dataset = ImageFolder(dataset_path + '/val', transform=val_test_transform)
    test_dataset = ImageFolder(dataset_path + '/test', transform=val_test_transform)

    # 動態取得類別數量與名稱
    class_names = train_dataset.classes
    num_classes = len(class_names)  # 自動根據資料夾數量設定

    print(f'Class = {num_classes}, {class_names}')
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=worker)

    # 2. 初始化 模型 (使用動態取得的 num_classes)
    if model_num == 0:  # MViTv2_tiny
        model = timm.create_model('mvitv2_tiny', pretrained=True, num_classes=num_classes, img_size=img_size)
    elif model_num == 1:  # MViTv2_small
        model = timm.create_model('mvitv2_small', pretrained=True, num_classes=num_classes, img_size=img_size)
    elif model_num == 2:  # MViTv2_base
        model = timm.create_model('mvitv2_base', pretrained=True, num_classes=num_classes, img_size=img_size)
    elif model_num == 3:  # MoblieNetV3_large
        model, backbone = Model_choose(class_names, 4)
    elif model_num == 4:  # MoblieNetV3_small
        model, backbone = Model_choose(class_names, 3)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-7)
    #optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=1E-4, weight_decay=1e-3)


    # 3. 初始化 LRFinder
    lr_finder = LRFinder(
        model,
        optimizer,
        criterion,
        device="cuda"
    )

    # 4. 執行 Range Test
    lr_finder.range_test(
        train_loader,
        val_loader=val_loader,  # 可選，用 val loss 更穩定
        start_lr=1e-7,
        end_lr=1e-1,
        num_iter=100,  # 迭代次數
        step_mode="exp",  # 指數遞增
        smooth_f=0.05,  # loss 平滑程度
        diverge_th=5.0  # 發散門檻
    )

    # 5. 畫圖 + 取得建議值
    ax, suggested_lr = lr_finder.plot(suggest_lr=True)

    print(f"建議學習率：{suggested_lr}")

    # 6. ⚠️ 重要！Reset 模型和 optimizer 回初始狀態
    lr_finder.reset()

    # 7. 用建議的 lr 重新訓練 (設定新的 optimizer)
    #optimizer = optim.AdamW(model.parameters(), lr=suggested_lr)
