import os
from PIL import Image

from ..registry import DATASOURCES
from .utils import McLoader


@DATASOURCES.register_module
class ImageList(object):

    def __init__(self, root, list_file, memcached, mclient_path):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.fns = [os.path.join(root, l.strip()) for l in lines]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        return img
