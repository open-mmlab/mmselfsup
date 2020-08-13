from ..registry import DATASOURCES
from .image_list import ImageList


@DATASOURCES.register_module
class Places205(ImageList):

    def __init__(self, root, list_file, memcached, mclient_path):
        super(Places205, self).__init__(
            root, list_file, memcached, mclient_path)
