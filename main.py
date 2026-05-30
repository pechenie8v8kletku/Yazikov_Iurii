
import os
from torch import nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import torch
from models import FirstModel,SecondModel,ThirdModel
from utils import save_epoch_checkpoint,save_only_model,seed_everything,train_epoch,validate_epoch,load_checkpoint
from Dataset import Point_face
from torchvision import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
seed_everything(67)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=64
LR=1e-4
NUM_WORKER=6
NUM_EPOCH=100
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

def train_val_split(meta_dict,ratio=0.2):
    items=list(meta_dict.items())
    val_size=int(len(items)*ratio)
    train_items,val_items=train_test_split(items,test_size=val_size,random_state=67)
    return dict(train_items),dict(val_items)
train_mepro_dict,val_mepro_dict=train_val_split(train_mepro,ratio=0.2)
train_300w_dict,val_300w_dict=train_val_split(train_300w,ratio=0.2)




dataset_train_300w=Point_face(train_300w_dict,transform=train_transform)
dataset_train_mepro=Point_face(train_mepro_dict,transform=train_transform)
dataset_val_300w=Point_face(val_300w_dict,transform=val_transform)
dataset_val_mepro=Point_face(val_mepro_dict,transform=val_transform)
dataset_test_300w=Point_face(test_300w,transform=val_transform)
dataset_test_mepro=Point_face(test_mepro,transform=val_transform)
dataset_train=ConcatDataset([dataset_train_300w,dataset_train_mepro])
dataset_test=ConcatDataset([dataset_test_300w,dataset_test_mepro])
dataset_val=ConcatDataset([dataset_val_300w,dataset_val_mepro])






if __name__=="__main__":
    train_loader=DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKER,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(dataset_val,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKER,pin_memory=True,persistent_workers=True)
    test_loader=DataLoader(dataset_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKER,pin_memory=True,persistent_workers=True)
    path = "runs"
    mod_path = "models"
    critrion=nn.MSELoss()


    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(mod_path):
        os.makedirs(mod_path)

    #RESNET18 backbone +MLP
    print("RESNET18 backbone +MLP+MSE")
    model_path="models/Resnet18_MLP_MSE"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter("runs/Resnet18_MLP_MSE")
    model1=FirstModel().to(DEVICE)
    optimizer=optim.AdamW([
        {
        "params": model1.encoder.parameters(),
        "lr":LR/5},
        {"params":model1.head.parameters(),
         "lr":LR
        }
    ],weight_decay=WEIGHT_DECAY)

    best_metric = float("inf")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=4)
    for epoch in range(NUM_EPOCH):
         train_loss,train_metric=train_epoch(model=model1,loader=train_loader,criterion=critrion,optimizer=optimizer,device=DEVICE,epoch=epoch,writer=writer)
         val_loss,vall_metric=validate_epoch(model=model1,loader=val_loader,criterion=critrion,device=DEVICE,epoch=epoch,writer=writer)
         print("epoch:", epoch, "train loss:", train_loss, "val loss:",val_loss,"train_metric: ",train_metric,"val_metric:",vall_metric)
         scheduler.step(val_loss)
         save_only_model(model1,epoch=epoch,path="models/Resnet18_MLP_MSE/")
         if vall_metric < best_metric:
            best_metric = vall_metric
            torch.save(
            {"epoch": epoch,"metric": vall_metric,"model": model1.state_dict()},
            "models/Resnet18_MLP_MSE/best_model.pt"
            )



    # EfficientNet backbone +MLP
    print("EfficientNet backbone +MLp+MSE")
    model_path = "models/Efficent_net_b2_MLP_MSE"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter("runs/Efficent_net_b2_MLP_MSE")
    model2 = SecondModel().to(DEVICE)
    optimizer = optim.AdamW([
        {
            "params": model2.encoder.parameters(),
            "lr": LR / 5},
        {"params": model2.head.parameters(),
         "lr": LR
         }
    ], weight_decay=WEIGHT_DECAY)
    best_metric = float("inf")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    for epoch in range(NUM_EPOCH):
        train_loss,train_metric = train_epoch(model=model2,loader=train_loader,criterion=critrion,optimizer=optimizer,device=DEVICE,epoch=epoch,writer=writer)
        val_loss,vall_metric = validate_epoch(model=model2,loader=val_loader,criterion=critrion,device=DEVICE,epoch=epoch,writer=writer)
        print("epoch:", epoch, "train loss:", train_loss, "val loss:",val_loss,"train_metric: ",train_metric,"val_metric:",vall_metric)
        scheduler.step(val_loss)
        save_only_model(model2, epoch=epoch, path="models/Efficent_net_b2_MLP_MSE/")
        if vall_metric < best_metric:
            best_metric = vall_metric
            torch.save(
                {"epoch": epoch, "metric": vall_metric, "model": model2.state_dict()},
                "models/Efficent_net_b2_MLP_MSE/best_model.pt"
            )

    # MobileNETV3 backbone +mlp
    print("MobileNETV3 backbone +mlp+MSE")
    model_path = "models/MobileNETV3_MLP_MSE"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter("runs/MobileNETV3_MLP_MSE")
    model3 = ThirdModel().to(DEVICE)
    optimizer = optim.AdamW([
        {
            "params": model3.encoder.parameters(),
            "lr": LR / 5},
        {"params": model3.head.parameters(),
         "lr": LR
         }
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    best_metric = float("inf")
    for epoch in range(NUM_EPOCH):
        train_loss,train_metric = train_epoch(model=model3,loader=train_loader,criterion=critrion,optimizer=optimizer,device=DEVICE,epoch=epoch,writer=writer)
        val_loss,vall_metric = validate_epoch(model=model3,loader=val_loader,criterion=critrion,device=DEVICE,epoch=epoch,writer=writer)
        print("epoch:", epoch, "train loss:", train_loss, "val loss:",val_loss,"train_metric: ",train_metric,"val_metric:",vall_metric)
        scheduler.step(val_loss)
        save_only_model(model3, epoch=epoch, path="models/MobileNETV3_MLP_MSE/")
        if vall_metric < best_metric:
            best_metric = vall_metric
            torch.save(
                {"epoch": epoch, "metric": vall_metric, "model": model3.state_dict()},
                "models/Efficent_net_b2_MLP_MSE/best_model.pt"
            )




