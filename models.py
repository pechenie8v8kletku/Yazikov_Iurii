import torch
import torch.nn as nn
import timm
from torchvision.models import resnet18,ResNet18_Weights,efficientnet_b2,EfficientNet_B2_Weights,MobileNet_V3_Large_Weights,mobilenet_v3_large,convnext_tiny,ConvNeXt_Tiny_Weights,efficientnet_b3,EfficientNet_B3_Weights

# в этом файле описаны модели которые были использованы для ресерча, из них только tiny convnext не проходит по условию меньше 60 мб остальные проходят, также все модели
# были взяты предобученными в среднем с учетом ограничения в 60 мб наиболее оптимальной идей для улучшения качества будет в дальнейшем уже не перебор архитектур
# а выбор Loss функции которая будет наилучшим образом учитывать специфику задачи, также возможно стоит почистить датасет от выбросов,для этого к примеру можно обучить модель сначала на одном потом на другом датасете и отсмотреть изобржаения с самым большим отклоеннием
# Также при просмотре наложенных точек на изображение было обнаружено что точки наложены не идеально.
# RESNet 18 + MLP
class FirstModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(FirstModel, self).__init__()
        modelres = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modelres.conv1,
            modelres.bn1,
            modelres.relu,
            modelres.maxpool,
            modelres.layer1,
            modelres.layer2,
            modelres.layer3,
            modelres.layer4,
            modelres.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x=self.encoder(x)
        x=torch.flatten(x, 1)
        x=self.head(x)
        return x
# Efficent net B2
class SecondModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(SecondModel, self).__init__()
        modeleff = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modeleff.features,
            modeleff.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )


    def forward(self, x):
        x = self.encoder(x)
        x=torch.flatten(x, 1)
        x=self.head(x)
        return x



# Mobbilente V3 large
class ThirdModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(ThirdModel, self).__init__()
        modelnet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            modelnet.features,
            modelnet.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
# RESNet 18 + MLP+ no avgpool
class FourthModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(FourthModel, self).__init__()
        modelres = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modelres.conv1,
            modelres.bn1,
            modelres.relu,
            modelres.maxpool,
            modelres.layer1,
            modelres.layer2,
            modelres.layer3,
            modelres.layer4
        )
        self.head = nn.Sequential(
            nn.Linear(7*7*512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x=self.encoder(x)
        x=torch.flatten(x, 1)
        x=self.head(x)
        return x
# ConvNExt
class FifthModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(FifthModel, self).__init__()
        modelres = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modelres.features,
            modelres.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

#mobileVIT
class SixthModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(SixthModel, self).__init__()
        modelvit= timm.create_model(
            "mobilevit_s",
            pretrained=True,num_classes=0
        )
        self.encoder = modelvit

        self.head = nn.Sequential(
            nn.Linear(640, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

# Efficent net B3
class SeventhModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(SeventhModel, self).__init__()
        modeleff = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modeleff.features,
            modeleff.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x