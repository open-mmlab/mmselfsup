from mmselfsup.models import MAE
from mmselfsup.registry import MODELS


@MODELS.register_module()
class DummyMAE(MAE):
    """Implements a dummy wrapper for demonstration purpose.

    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('*************************\n'
              '* Welcome to MMSelfSup! *\n'
              '*************************')
        super().__init__(**kwargs)
