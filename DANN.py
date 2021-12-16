import torch
import torch.nn as nn

from urllib.request import urlretrieve

from os.path import exists


ALEXNET_PRETRAINED = "alexnet-owt-4df8aa71.pth"
ALEXNET_PRETRAINED_URL = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"


class DANN(nn.Module):
    def __init__(
        self, num_classes: int = 1000, num_domains: int = 2, dropout: float = 0.5
    ) -> None:
        super().__init__()

        # "core" layers that extract the domain-invariant features
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

        # original AlexNet classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(num_classes: int, num_domains: int, pretrained: bool) -> DANN:
    model = DANN()

    if pretrained:
        if not exists(ALEXNET_PRETRAINED):
            print("Downloading the AlexNet weights... ", end="")
            urlretrieve(ALEXNET_PRETRAINED_URL, filename=ALEXNET_PRETRAINED)
            print("Done!")

        model.load_state_dict(torch.load(ALEXNET_PRETRAINED), strict=False)

        # copy the weights of the linear layers (1st and 4th entries)
        for i in [1, 4]:
            model.domain_classifier[i].weight.data = model.classifier[i].weight.data
            model.domain_classifier[i].bias.data = model.classifier[i].bias.data
        
        # restore the fully connected layers (TODO: is it really necessary?)
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.domain_classifier[6] = nn.Linear(4096, num_domains)

        # freeze the features layer 
        for layer in model.features:
            layer.requires_grad_ = False

    return model


if __name__ == "__main__":
    NUM_CLASSES = 7
    NUM_DOMAINS = 2

    model = build_model(NUM_CLASSES, NUM_DOMAINS, True)
    print(model)
