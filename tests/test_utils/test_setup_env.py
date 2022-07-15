# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmselfsup.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmselfsup.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmselfsup.datasets', None)
        sys.modules.pop('mmselfsup.datasets.places205', None)
        DATASETS._module_dict.pop('Places205', None)
        self.assertFalse('Places205' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('Places205' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmselfsup.datasets')
        sys.modules.pop('mmselfsup.datasets.places205')
        DATASETS._module_dict.pop('Places205', None)
        self.assertFalse('Places205' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('Places205' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmselfsup')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning,
                'The current default scope "test" is not "mmselfsup"'):
            register_all_modules(init_default_scope=True)
