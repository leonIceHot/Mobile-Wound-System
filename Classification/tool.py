import os
import torch
import torchvision.models as models
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
# from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_fscore_support, average_precision_score
# from tqdm import tqdm
# from IPython.display import clear_output
import pandas as pd
import torch
from tqdm import tqdm
import timm
from util.contrastive_loss import ClsWithProj, SupConModel

global model_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(model_num):
    if model_num == 0:      #   ResNet 18
        model = models.resnet18(pretrained = True)  #pretrained 決定是否載入預訓練權重
    elif model_num == 1:    #   ResNet 18
        model = models.resnet50(pretrained = True)
    elif model_num == 2:    #   GhostNet
        model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)

    model_ = model
    return model

def Create_NF(path):

    if os.path.exists(path):
        os.mkdir(path)
    '''
    if not os.path.isdir(path):
        #os.makedirs(path)
        os.makedirs(path,exist_ok=True)
    '''
def Create_New_File(path):  #path: '.\\' + model_name + date + '\\weight'
    #Create_NF(path)

    weights_dir = path + 'weights'
    #print(weights_dir)

    results_dir = path + 'results' #results_dir的值為要建立的資料夾名稱
    #print(results_dir)

    checkpoint_dir = path + 'checkpoints'
    #print(checkpoint_dir)

    Create_NF(weights_dir)
    Create_NF(results_dir)
    Create_NF(checkpoint_dir)

    return weights_dir, results_dir, checkpoint_dir

def Create_csv(path, file_name, used):
    log_file = os.path.join(path, file_name)

    # Initialize an empty DataFrame to store logs
    if used == 'train_result':
        log_df = pd.DataFrame(columns = ['Epoch', 'Training Loss', 'Training Accuracy', 'Training time', 'Validation Loss', 'Validation Accuracy', 'Validation time', 'Learning Rate'])

    elif used == 'final_result':
        log_df = pd.DataFrame(columns = ['Number', 'Testing time', '', 'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', 'Best loss', 'Testing mean time'])
    
    return log_file, log_df

def Model_choose(class_names, model_num):
    num_classes = len(class_names) #類別數量
    #=================================================================================================
    if model_num == 0:      #   ResNet 18
        model = models.resnet18(pretrained = True).to(device)  #pretrained 決定是否載入預訓練權重
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for (name, param) in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        backbone = nn.Sequential(*list(model.children())[:-1])  #backbone(img) 的輸出 shape 為 [B, 512, 1, 1]，再 .view(B, -1) 就是 [B, 512] 特徵向量

    #=================================================================================================
    elif model_num == 1:    #   ResNet 50
        model = models.resnet50(pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for (name, param) in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        backbone = nn.Sequential(*list(model.children())[:-1])

    #=================================================================================================
    elif model_num == 2:    #   GhossyNet_v3_100
        model = timm.create_model('ghostnet_100', pretrained = True, num_classes = num_classes)
        in_features = model.classifier.in_features
        #   凍結所有參數
        for (name, param) in model.named_parameters():
            param.requires_grad = False
        model.classifier = torch.nn.Linear(in_features = in_features, out_features = num_classes)
        #   解凍分類層
        for param in model.classifier.parameters():
            param.requires_grad = True
        backbone = nn.Sequential(*list(model.children())[:-1])
    
    #=================================================================================================
    elif model_num == 3:    #    MoblieNetV3_small
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        for (name, param) in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        backbone = nn.Sequential(*list(model.children())[:-1])
    
    #=================================================================================================
    elif model_num == 4:    #   MoblieNetV3_large
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        for (name, param) in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        backbone = nn.Sequential(*list(model.children())[:-1])

    return model, backbone

def Model_choose_contrastive(class_names, model_num):
    num_classes = len(class_names) #類別數量
    #=================================================================================================
    if model_num == 0:      #   ResNet 18
        backbone = models.resnet18(pretrained = True)  #pretrained 決定是否載入預訓練權重
        for (name, param) in backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model = SupConModel(backbone, feat_dim=128, out_features=num_classes).cuda()

    #=================================================================================================
    elif model_num == 1:    #   ResNet 50
        backbone = models.resnet50(pretrained = True)
        for (name, param) in backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model = SupConModel(backbone, feat_dim=128, out_features=num_classes).cuda()

    #=================================================================================================
    elif model_num == 2:    #   GhostNet_v3_100
        backbone = timm.create_model('ghostnet_100', pretrained = True, num_classes = 0)
        for (name, param) in backbone.named_parameters():
            param.requires_grad = False

        model = SupConModel(backbone, feat_dim=128, out_features=num_classes).cuda()
        #   解凍分類層
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    #=================================================================================================
    elif model_num == 3:    #    MoblieNetV3_small
        num_classes = 5
        backbone = models.mobilenet_v3_small(pretrained=True)
        for (name, param) in backbone.named_parameters():
            if 'classifier' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        backbone.classifier = nn.Identity()
        model = SupConModel(backbone, feat_dim=128, out_features=num_classes).cuda()

    #=================================================================================================
    elif model_num == 4:    #   MoblieNetV3_large
        backbone = models.mobilenet_v3_large(pretrained=True)
        for (name, param) in backbone.named_parameters():
            if 'classifier' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        backbone.classifier = nn.Identity()
        model = SupConModel(backbone, feat_dim=128, out_features=num_classes).cuda()
    #=================================================================================================

    return model, backbone

def cls_check_checkpoint(model, optimizer, scheduler = None, weight_path = '', str_ = '', Checkpoint_path = '', data_name = ''):
    if os.path.exists(f"{Checkpoint_path}{model}_{data_name}_checkpoint_{str_}.tar"):
        checkpoint = torch.load(f"{Checkpoint_path}{model}_{data_name}_checkpoint_{str_}.tar")
        weight_path = checkpoint["weight_path"]
        print(f"weight_path: {weight_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler != None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_continue = checkpoint["epoch"]
        best_loss = checkpoint["best_loss_score"]
        best_acc = checkpoint["best_acc_score"]
        loss_list = checkpoint["matrics_loss"]
        acc_list = checkpoint["matrics_acc"]
        if "train_time" in checkpoint:
            train_time = checkpoint["train_time"]
        else: train_time = [0]
        if "val_time" in checkpoint:
            val_time = checkpoint["val_time"]
        else: val_time = [0]
        print(f"take last checkpoint:")
        return model, optimizer, scheduler, epoch_continue, best_loss, best_acc, weight_path, loss_list, acc_list, train_time, val_time
    else: 
        return model, optimizer, scheduler, 0, 100, 0, weight_path, [], [], [0], [0]
    
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
    # 
    # else:
    #     checkpoint = {
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'epoch': epoch,
    #         'best_val_accuracy': best_val_accuracy,
    #         'train_loss': train_loss,
    #         'train_acc': train_acc,
    #         'val_loss': val_loss,
    #         'val_acc': val_acc
    #     }
    #     if scheduler and hasattr(scheduler, 'state_dict'):
    #         checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    #     torch.save(checkpoint, f'{save_dir}/checkpoint_path')
    #     print("checkpoint saved.")

def New_str_loader_img_Y_TF(img1, lbl):
    # 轉換 PyTorch 張量為 NumPy 數組
    img1 = img1.cpu().numpy().transpose((1, 2, 0))
    print(np.shape(img1))
    # 顯示圖像和分割圖
    plt.title(f'{lbl}')
    plt.imshow(img1)

def Check_loss_function(path):
    files = os.listdir(path)
    file_count = len(files) #確認資料夾中檔案數量

    filename_parts = [item.split('_') for item in files]    #以'_"分割list中字串
    # print(filename_parts)

    choose_best_loss = '100.0'
    best_list = 0
    for i in range(file_count):     #選擇資料夾中Loss最低的作為Best model

        if filename_parts[i][2] < choose_best_loss:
            choose_best_loss = filename_parts[i][2]
            best_list = i
        else:
            choose_best_loss = choose_best_loss
        choose_best_file = files[best_list]
    print(f'The best model is {choose_best_file}')

    model_.load_state_dict(torch.load(f'{path}\\{choose_best_file}')) #Load the best model for testing and confusion matrix