import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

def load_model(code_length):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.

    Returns
        model (torch.nn.Module): CNN model.
    """
    model = AlexNet(code_length)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
    model.load_state_dict(state_dict, strict=False)

    return model

class AlexNet(nn.Module):

    def __init__(self, code_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

        self.classifier = self.classifier[:-1]
        # self.hash_layer = nn.Linear(4096, code_length)
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Tanh(),
        )
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        
    def forward(self, x, out_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        feat = self.classifier(x)
        x = self.hash_layer(feat)
        if out_features:
            return x, feat
        else:
            return x