# Copyright (c) OpenMMLab. All rights reserved.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MMSelfSup'
copyright = '2020-2021, OpenMMLab'
author = 'MMSelfSup Authors'

# The full version, including alpha/beta/rc tags
version_file = '../../mmselfsup/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
    'sphinx_markdown_tables', 'sphinx_copybutton', 'myst_parser'
]

autodoc_mock_imports = ['json_tricks', 'mmselfsup.version']

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmselfsup'
        },
        {
            'name':
            '版本',
            'children': [
                {
                    'name': 'MMSelfSup 0.x',
                    'url': 'https://mmselfsup.readthedocs.io/zh_CN/latest/',
                    'description': 'main 分支文档'
                },
                {
                    'name': 'MMSelfSup 1.x',
                    'url': 'https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/',
                    'description': '1.x 分支文档'
                },
            ],
            'active':
            True,
        },
    ],
    'menu_lang':
    'cn',
    'header_note': {
        'title':
        '注意',
        'content':
        '您正在阅读 MMSelfSup 0.x 版本的文档，而 MMSelfSup 0.x 版本将会在 2022 年末 '
        '开始逐步停止维护。我们建议您及时升级到 MMSelfSup 1.0.0rc 版本，享受由 '
        'OpenMMLab 2.0 带来的更多新特性和更佳的性能表现。阅读 MMSelfSup 1.0.0rc 的 '
        '<a href="https://github.com/open-mmlab/mmselfsup/releases">发版日志</a>, '  # noqa
        '<a href="https://github.com/open-mmlab/mmselfsup/tree/1.x">代码</a> '  # noqa
        '和 <a href="https://mmselfsup.readthedocs.io/en/dev-1.x/">文档</a> 获取更多信息。',  # noqa
    }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

language = 'zh_CN'

html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

# Enable ::: for my_st
myst_enable_extensions = ['colon_fence']
myst_heading_anchors = 4

master_doc = 'index'
