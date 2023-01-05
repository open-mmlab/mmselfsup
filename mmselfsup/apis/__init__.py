# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model
<<<<<<< HEAD

__all__ = ['init_model', 'inference_model']
=======
from .train import init_random_seed, set_random_seed, train_model

__all__ = [
    'init_random_seed', 'inference_model', 'set_random_seed', 'train_model',
    'init_model'
]
>>>>>>> upstream/master
