from PIL import Image

from torchvision.datasets import CIFAR10, CIFAR100

from ..registry import DATASOURCES


@DATASOURCES.register_module
class Cifar10(object):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    def __init__(self, root, split):
        assert split in ['train', 'test']
        try:
            self.cifar = CIFAR10(
                root=root, train=split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")
        self.labels = self.cifar.targets

    def get_length(self):
        return len(self.cifar)

    def get_sample(self, idx):
        img = Image.fromarray(self.cifar.data[idx])
        target = self.labels[idx]  # img: HWC, RGB
        return img, target


@DATASOURCES.register_module
class Cifar100(object):

    CLASSES = None

    def __init__(self, root, split):
        assert split in ['train', 'test']
        try:
            self.cifar = CIFAR100(
                root=root, train=spilt == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")
        self.labels = self.cifar.targets

    def get_sample(self, idx):
        img = Image.fromarray(self.cifar.data[idx])
        target = self.labels[idx]  # img: HWC, RGB
        return img, target
