from PIL import Image
from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class DeepClusterDataset(BaseDataset):
    """Dataset for DC and ODC.
    """

    def __init__(self, data_source, pipeline):
        super(DeepClusterDataset, self).__init__(data_source, pipeline)
        # init clustering labels
        self.labels = [-1 for _ in range(self.data_source.get_length())]

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        label = self.labels[idx]
        img = self.pipeline(img)
        return dict(img=img, pseudo_label=label, idx=idx)

    def assign_labels(self, labels):
        assert len(self.labels) == len(labels), \
            "Inconsistent lenght of asigned labels, \
            {} vs {}".format(len(self.labels), len(labels))
        self.labels = labels[:]

    def evaluate(self, scores, keyword, logger=None):

        raise NotImplemented
