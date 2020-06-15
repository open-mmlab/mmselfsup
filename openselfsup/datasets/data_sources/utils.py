import io
from PIL import Image
try:
    import mc
except ImportError as E:
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


class McLoader(object):

    def __init__(self, mclient_path):
        assert mclient_path is not None, \
            "Please specify 'data_mclient_path' in the config."
        self.mclient_path = mclient_path
        server_list_config_file = "{}/server_list.conf".format(
            self.mclient_path)
        client_config_file = "{}/client.conf".format(self.mclient_path)
        self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                      client_config_file)

    def __call__(self, fn):
        try:
            img_value = mc.pyvector()
            self.mclient.Get(fn, img_value)
            img_value_str = mc.ConvertBuffer(img_value)
            img = pil_loader(img_value_str)
        except:
            print('Read image failed ({})'.format(fn))
            return None
        else:
            return img
