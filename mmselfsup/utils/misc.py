# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModel, is_model_wrapper


def get_model(model) -> BaseModel:
    """Get model if the input model is a model wrapper.

    Args:
        model (BaseModel): A model may be a model wrapper.

    Returns:
        BaseModel: The model without model wrapper.
    """
    if is_model_wrapper(model):
        return model.module
    else:
        return model
