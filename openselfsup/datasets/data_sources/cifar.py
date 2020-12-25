from abc import ABCMeta, abstractmethod
from PIL import Image

from torchvision.datasets import CIFAR10, CIFAR100

from ..registry import DATASOURCES


class Cifar(metaclass=ABCMeta):

    CLASSES = None

    def __init__(self, root, split, return_label=True):
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.return_label = return_label
        self.cifar = None
        self.set_cifar()
        self.labels = self.cifar.targets

    @abstractmethod
    def set_cifar(self):
        pass

    def get_length(self):
        return len(self.cifar)

    def get_sample(self, idx):
        img = Image.fromarray(self.cifar.data[idx])
        if self.return_label:
            target = self.labels[idx]  # img: HWC, RGB
            return img, target
        else:
            return img


@DATASOURCES.register_module
class Cifar10(Cifar):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)

    def set_cifar(self):
        try:
            self.cifar = CIFAR10(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")


@DATASOURCES.register_module
class Cifar100(Cifar):

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)

    def set_cifar(self):
        try:
            self.cifar = CIFAR100(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")
