import gc
import psutil
import numpy as np
import os
from torch import nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import torch
from models import FirstModel, SecondModel, ThirdModel, FourthModel,FifthModel,SixthModel,SeventhModel
from utils import save_epoch_checkpoint,save_only_model,seed_everything,train_epoch,validate_epoch,load_checkpoint
from Dataset import Point_face
from torchvision import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# Гиперпараметры
seed_everything(67)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=64
LR=5e-4
NUM_WORKER=6
NUM_EPOCH=160
WEIGHT_DECAY=1e-4
# PATHES to metas
Meta_300w_test="meta_test_300w.json"
Meta_300w_train="meta_train_300w.json"
Meta_mepro_test="meta_test_mepro.json"
Meta_mepro_train="meta_train_mepro.json"

# Augmentations
train_transform = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

# Dataset+loader
with open(Meta_300w_test,'r') as f:
    test_300w=json.load(f)
with open(Meta_300w_train,'r') as f:
    train_300w=json.load(f)
with open(Meta_mepro_test,'r') as f:
    test_mepro=json.load(f)
with open(Meta_mepro_train,'r') as f:
    train_mepro=json.load(f)
# функция для сплита на вал и трейн трейн части
def train_val_split(meta_dict,ratio=0.2):
    items=list(meta_dict.items())
    val_size=int(len(items)*ratio)
    train_items,val_items=train_test_split(items,test_size=val_size,random_state=67)
    return dict(train_items),dict(val_items)
train_mepro_dict,val_mepro_dict=train_val_split(train_mepro,ratio=0.2)
train_300w_dict,val_300w_dict=train_val_split(train_300w,ratio=0.2)



#инициализация датасетов
dataset_train_300w=Point_face(train_300w_dict,transform=train_transform)
dataset_train_mepro=Point_face(train_mepro_dict,transform=train_transform)
dataset_val_300w=Point_face(val_300w_dict,transform=val_transform)
dataset_val_mepro=Point_face(val_mepro_dict,transform=val_transform)
dataset_test_300w=Point_face(test_300w,transform=val_transform)
dataset_test_mepro=Point_face(test_mepro,transform=val_transform)
dataset_train=ConcatDataset([dataset_train_300w,dataset_train_mepro])
dataset_test=ConcatDataset([dataset_test_300w,dataset_test_mepro])
dataset_val=ConcatDataset([dataset_val_300w,dataset_val_mepro])

# кастомная лосс функция, основная идея, части точек обладают некоторой связностью которой не стоит принебрегать, по опытным результатам улучшило перформанс по метрике отклонений
# за основу берется MSE к нему добавляется средний квадрат разница смещенности относительно соседей точки из предсказания и из истинной разметки, в теории лучше ориентирует не толкьо на точку
# но и еще на структуру
class GraphLaplacianLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), alpha=0.2):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha

        A = torch.zeros((68, 68), dtype=torch.float32)

        def add_chain(indices):
            for i in range(len(indices) - 1):
                u, v = indices[i], indices[i + 1]
                A[u, v] = 1.0
                A[v, u] = 1.0

        def add_loop(indices):
            add_chain(indices)
            A[indices[-1], indices[0]] = 1.0
            A[indices[0], indices[-1]] = 1.0

        add_chain(list(range(0, 17)))
        add_chain(list(range(17, 22)))
        add_chain(list(range(22, 27)))
        add_chain(list(range(27, 31)))
        add_chain(list(range(31, 36)))
        add_loop(list(range(36, 42)))
        add_loop(list(range(42, 48)))
        add_loop(list(range(48, 60)))
        add_loop(list(range(60, 68)))

        D = torch.diag_embed(A.sum(dim=1))
        L = D - A
        self.register_buffer("L", L)

    def forward(self, predictions, targets):
        base = self.base_loss(predictions, targets)
        preds_3d = predictions.view(-1, 68, 2)
        targets_3d = targets.view(-1, 68, 2)

        pred_lap = torch.matmul(self.L, preds_3d)
        target_lap = torch.matmul(self.L, targets_3d)

        lap_loss = torch.mean((pred_lap - target_lap) ** 2)

        return base + self.alpha  * lap_loss



# функция реализующая полное обучение модели
def train_model(model,train_loader,val_loader,criterion,model_name):
    print(model_name)
    critrion=criterion
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter(f"runs/{model_name}")
    model = model().to(DEVICE)
    optimizer = optim.AdamW([
        {
            "params": model.encoder.parameters(),
            "lr": LR / 5},
        {"params": model.head.parameters(),
         "lr": LR
         }
    ], weight_decay=WEIGHT_DECAY)

    best_metric = float("inf")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    for epoch in range(NUM_EPOCH):

        print(
            "RAM:",
            round(
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3,
                2
            ),
            "GB"
        )
        train_loss, train_metric = train_epoch(model=model, loader=train_loader, criterion=critrion,optimizer=optimizer, device=DEVICE, epoch=epoch, writer=writer)
        val_loss, vall_metric = validate_epoch(model=model, loader=val_loader, criterion=critrion, device=DEVICE,epoch=epoch, writer=writer)
        print("epoch:", epoch, "train loss:", train_loss, "val loss:", val_loss, "train_metric: ", train_metric,"val_metric:", vall_metric)
        scheduler.step(val_loss)
        save_only_model(model, epoch=epoch, path=f"models/{model_name}/")
        if vall_metric < best_metric:
            best_metric = vall_metric
            torch.save(
                {"epoch": epoch, "metric": vall_metric, "model": model.state_dict()},
                f"models/{model_name}/best_model.pt"
            )
    writer.close()
    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()


if __name__=="__main__":
    train_loader=DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKER,persistent_workers=True,pin_memory=False)
    val_loader=DataLoader(dataset_val,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKER,persistent_workers=True,pin_memory=False)
    path = "runs"
    mod_path = "models"
    critrion=GraphLaplacianLoss(base_loss=nn.SmoothL1Loss(),alpha=0.07).to(DEVICE)


    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(mod_path):
        os.makedirs(mod_path)
    #train_model(model=FirstModel,train_loader=train_loader,val_loader=val_loader,criterion=critrion,model_name="Resnet18_GraphLaplacian_avgpool")
    #train_model(model=FifthModel,train_loader=train_loader,val_loader=val_loader,criterion=critrion,model_name="ConvNeXt_tiny_GraphLaplacian_avgpool")
    #train_model(model=SecondModel,train_loader=train_loader,val_loader=val_loader,criterion=critrion,model_name="EfficientNet_B2_GraphLaplacian_avgpool")
    train_model(model=SeventhModel,train_loader=train_loader,val_loader=val_loader,criterion=critrion,model_name="EfficientNet_B3_GraphLaplacian+smoothL1_trying_fit_alpha_avgpool")

    # #RESNET18 backbone +MLP
    # print("RESNET18 backbone +MLP+Smooth_L1")
    # model_path="models/Resnet18_MLP_MSE"
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # writer = SummaryWriter("runs/Resnet18_MLP_MSE")
    # model1=FirstModel().to(DEVICE)
    # optimizer=optim.AdamW([
    #     {
    #     "params": model1.encoder.parameters(),
    #     "lr":LR/5},
    #     {"params":model1.head.parameters(),
    #      "lr":LR
    #     }
    # ],weight_decay=WEIGHT_DECAY)
    #
    # best_metric = float("inf")
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=4)
    # for epoch in range(NUM_EPOCH):
    #      train_loss,train_metric=train_epoch(model=model1,loader=train_loader,criterion=critrion,optimizer=optimizer,device=DEVICE,epoch=epoch,writer=writer)
    #      val_loss,vall_metric=validate_epoch(model=model1,loader=val_loader,criterion=critrion,device=DEVICE,epoch=epoch,writer=writer)
    #      print("epoch:", epoch, "train loss:", train_loss, "val loss:",val_loss,"train_metric: ",train_metric,"val_metric:",vall_metric)
    #      scheduler.step(val_loss)
    #      save_only_model(model1,epoch=epoch,path="models/Resnet18_MLP_MSE/")
    #      if vall_metric < best_metric:
    #         best_metric = vall_metric
    #         torch.save(
    #         {"epoch": epoch,"metric": vall_metric,"model": model1.state_dict()},
    #         "models/Resnet18_MLP_MSE/best_model.pt"
    #         )
    #
    #
    #
    # # EfficientNet backbone +MLP
    # print("EfficientNet backbone +MLp+MSE")
    # model_path = "models/Efficent_net_b2_MLP_MSE"
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # writer = SummaryWriter("runs/Efficent_net_b2_MLP_MSE")
    # model2 = SecondModel().to(DEVICE)
    # optimizer = optim.AdamW([
    #     {
    #         "params": model2.encoder.parameters(),
    #         "lr": LR / 5},
    #     {"params": model2.head.parameters(),
    #      "lr": LR
    #      }
    # ], weight_decay=WEIGHT_DECAY)
    # best_metric = float("inf")
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    # for epoch in range(NUM_EPOCH):
    #     train_loss,train_metric = train_epoch(model=model2,loader=train_loader,criterion=critrion,optimizer=optimizer,device=DEVICE,epoch=epoch,writer=writer)
    #     val_loss,vall_metric = validate_epoch(model=model2,loader=val_loader,criterion=critrion,device=DEVICE,epoch=epoch,writer=writer)
    #     print("epoch:", epoch, "train loss:", train_loss, "val loss:",val_loss,"train_metric: ",train_metric,"val_metric:",vall_metric)
    #     scheduler.step(val_loss)
    #     save_only_model(model2, epoch=epoch, path="models/Efficent_net_b2_MLP_MSE/")
    #     if vall_metric < best_metric:
    #         best_metric = vall_metric
    #         torch.save(
    #             {"epoch": epoch, "metric": vall_metric, "model": model2.state_dict()},
    #             "models/Efficent_net_b2_MLP_MSE/best_model.pt"
    #         )
    #
    # # MobileNETV3 backbone +mlp
    # print("MobileNETV3 backbone +mlp+MSE")
    # model_path = "models/MobileNETV3_MLP_MSE"
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # writer = SummaryWriter("runs/MobileNETV3_MLP_MSE")
    # model3 = ThirdModel().to(DEVICE)
    # optimizer = optim.AdamW([
    #     {
    #         "params": model3.encoder.parameters(),
    #         "lr": LR / 5},
    #     {"params": model3.head.parameters(),
    #      "lr": LR
    #      }
    # ], weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    # best_metric = float("inf")
    # for epoch in range(NUM_EPOCH):
    #     train_loss,train_metric = train_epoch(model=model3,loader=train_loader,criterion=critrion,optimizer=optimizer,device=DEVICE,epoch=epoch,writer=writer)
    #     val_loss,vall_metric = validate_epoch(model=model3,loader=val_loader,criterion=critrion,device=DEVICE,epoch=epoch,writer=writer)
    #     print("epoch:", epoch, "train loss:", train_loss, "val loss:",val_loss,"train_metric: ",train_metric,"val_metric:",vall_metric)
    #     scheduler.step(val_loss)
    #     save_only_model(model3, epoch=epoch, path="models/MobileNETV3_MLP_MSE/")
    #     if vall_metric < best_metric:
    #         best_metric = vall_metric
    #         torch.save(
    #             {"epoch": epoch, "metric": vall_metric, "model": model3.state_dict()},
    #             "models/Efficent_net_b2_MLP_MSE/best_model.pt"
    #         )




