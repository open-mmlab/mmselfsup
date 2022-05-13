# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmselfsup.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmselfsup.registry import TRANSFORMS

        # not init default scope
        sys.modules.pop('mmselfsup.datasets.pipelines', None)
        sys.modules.pop('mmselfsup.datasets.pipelines.formatting', None)
        TRANSFORMS._module_dict.pop('PackSelfSupInputs', None)
        self.assertFalse('PackSelfSupInputs' in TRANSFORMS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('PackSelfSupInputs' in TRANSFORMS.module_dict)

        # init default scope
        sys.modules.pop('mmselfsup.datasets.pipelines')
        sys.modules.pop('mmselfsup.datasets.pipelines.formatting')
        TRANSFORMS._module_dict.pop('PackSelfSupInputs', None)
        self.assertFalse('PackSelfSupInputs' in TRANSFORMS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('PackSelfSupInputs' in TRANSFORMS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmselfsup')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning,
                'The current default scope "test" is not "mmselfsup"'):
            register_all_modules(init_default_scope=True)
