from torch.autograd import Function
from torchvision.datasets import VisionDataset

import os.path
from os.path import join

from .utils import pil_loader

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha # save alpha for the backward step

        return x # x.view_as(x) ???

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None # why two return values ???


class PACSDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(PACSDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.filenames = []
        self.labels = []
        for root, dirs, files in os.walk(self.root, topdown=False):
            self.filenames += [root.split("/")[-1] + "/" + name for name in files]
            self.labels += list(dirs)

    def __getitem__(self, index):
        label, filename = self.filenames[index].split("/")

        image = pil_loader(join(self.root, label, filename))

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels.index(label)

    def __len__(self):
        """Return the number of files in the dataset"""
        return len(self.filenames)


if __name__ == "__main__":
    dataset = PACSDataset("data/photo")

    image, label = dataset.__getitem__(459)
    print("Image size:", image.size)
    print("Class label:", label)
