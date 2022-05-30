# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmselfsup.models.utils import Extractor


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)
        self.neck = nn.Identity()

    def forward(self, img, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_extractor():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))

    extract_dataloader = dict(
        batch_size=1,
        num_workers=1,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=test_dataset)

    # test init
    extractor = Extractor(
        extract_dataloader=extract_dataloader, dist_mode=False, pool_cfg=None)
    assert getattr(extractor, 'pool', None) is None

    # test init
    extractor = Extractor(
        extract_dataloader=extract_dataloader, dist_mode=False)

    # TODO: test runtime
    # As the BaseModel is not defined finally, I will add it later.

    # # test extractor
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     model = MMDataParallel(ExampleModel())
    #     optimizer = build_optimizer(model, optim_cfg)
    #     runner = build_runner(
    #         runner_cfg,
    #         default_args=dict(
    #             model=model,
    #             optimizer=optimizer,
    #             work_dir=tmpdir,
    #             logger=logging.getLogger()))
    #     features = extractor(runner)
    #     assert features.shape == (1, 1)
