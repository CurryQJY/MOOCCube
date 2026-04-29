import torch
import torch.nn as nn
import torchvision.models as models

class AnimalNet(nn.Module):
    def __init__(self, num_attrs=20, num_classes=7):
        super(AnimalNet, self).__init__()

        # 加载 ResNet18 主干网络
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features

        # 去掉原始分类层
        self.backbone.fc = nn.Identity()

        # 属性预测分支
        self.attr_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_attrs)
        )

        # 动物类别辅助分类分支
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 提取图像特征
        feat = self.backbone(x)

        # 输出属性 logits
        attr_logits = self.attr_head(feat)

        # 输出类别 logits
        cls_logits = self.cls_head(feat)

        return attr_logits, cls_logits