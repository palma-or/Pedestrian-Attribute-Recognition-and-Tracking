import torch
import torchvision.models as models
import torch.nn as nn

#resnet = models.resnet18(pretrained=True)   # ResNet18
resnet = models.resnet34(pretrained=True)   # ResNet34
backbone = nn.Sequential(*list(resnet.children())[:-1])   # remove output layer


# task-specific heads

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.fc = nn.Linear(resnet.fc.in_features, 1)   # output layer for binary gender classification

    def forward(self, x):
        x = self.fc(x)
        return x

class BagClassifier(nn.Module):
    def __init__(self):
        super(BagClassifier, self).__init__()
        self.fc = nn.Linear(resnet.fc.in_features, 1)   # output layer for binary bag classification

    def forward(self, x):
        x = self.fc(x)
        return x

class HatClassifier(nn.Module):
    def __init__(self):
        super(HatClassifier, self).__init__()
        self.fc = nn.Linear(resnet.fc.in_features, 1)   # output layer for binary hat classification

    def forward(self, x):
        x = self.fc(x)
        return x



# create model by combining ResNet backbone with task-specific heads

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.backbone = backbone
        self.gender_classifier = GenderClassifier()
        self.bag_classifier = BagClassifier()
        self.hat_classifier = HatClassifier()

    def forward(self, x):
        features = self.backbone(x).squeeze()   # extract features from ResNet backbone
        gender_output = self.gender_classifier(features)   # predict gender
        bag_output = self.bag_classifier(features)   # predict bag
        hat_output = self.hat_classifier(features)   # predict hat
        return gender_output, bag_output, hat_output
    
