# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmselfsup.models.utils import Extractor


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_lambda = 0.5
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        inputs, labels = [], []
        for x in data_batch:
            inputs.append(x['inputs'])
            labels.append(x['data_sample'])

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = torch.stack(inputs).to(device)
        labels = torch.stack(labels).to(device)
        outputs = self.linear(inputs)
        if return_loss:
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss, log_vars=dict(loss=loss.item()))
            return outputs
        else:
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


def test_extractor():
    dummy_dataset = DummyDataset()

    extract_dataloader = dict(
        batch_size=1,
        num_workers=1,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dummy_dataset)

    # test init
    extractor = Extractor(
        extract_dataloader=extract_dataloader, dist_mode=False, pool_cfg=None)
    assert getattr(extractor, 'pool', None) is None

    # test init
    extractor = Extractor(
        extract_dataloader=extract_dataloader,
        dist_mode=False,
        pool_cfg=dict(type='AvgPool2d', output_size=1))

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
