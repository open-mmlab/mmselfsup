#!/usr/bin/env python
import re
from collections import defaultdict
from pathlib import Path

from modelindex.load_model_index import load

MMSELFSUP_ROOT = Path(__file__).absolute().parents[2]
PAPERS_ROOT = Path('papers')  # Path to save generated paper pages.
GITHUB_PREFIX = 'https://github.com/open-mmlab/mmselfsup/blob/1.x/'
MODELZOO_TEMPLATE = """
# Model Zoo Statistics

* Number of papers: {num_papers}
{type_msg}

* Number of checkpoints: {num_ckpts}
{paper_msg}
"""

model_index = load(str(MMSELFSUP_ROOT / 'model-index.yml'))


def build_collections(model_index):
    col_by_name = {}
    for col in model_index.collections:
        setattr(col, 'models', [])
        col_by_name[col.name] = col

    for model in model_index.models:
        col = col_by_name[model.in_collection]
        col.models.append(model)
        setattr(model, 'collection', col)


build_collections(model_index)


def count_papers(model_index):
    ckpt_dict = dict()
    type_count = defaultdict(int)
    paper_msgs = []

    for model in model_index.models:
        if model.collection.name in ckpt_dict.keys():
            if model.weights:
                ckpt_dict[model.collection.name] += 1
        else:
            ckpt_dict[model.collection.name] = 1

        downstream_info = model.data.get('Downstream', [])
        for downstream_task in downstream_info:
            if downstream_task.get('Weights', None):
                ckpt_dict[model.collection.name] += 1

    for collection in model_index.collections:
        name = collection.name
        title = collection.paper['Title']
        papertype = collection.data.get('type', 'Algorithm')
        type_count[papertype] += 1

        with open(MMSELFSUP_ROOT / collection.readme) as f:
            readme = f.read()
        readme = PAPERS_ROOT / Path(
            collection.filepath).parent.with_suffix('.md').name
        paper_msgs.append(
            f'\t- [{papertype}] [{title}]({readme}) ({ckpt_dict[name]} '
            f'ckpts)')

    type_msg = '\n'.join(
        [f'\t- {type_}: {count}' for type_, count in type_count.items()])
    paper_msg = '\n'.join(paper_msgs)

    modelzoo = MODELZOO_TEMPLATE.format(
        num_papers=sum(type_count.values()),
        num_ckpts=sum(ckpt_dict.values()),
        type_msg=type_msg,
        paper_msg=paper_msg,
    )

    with open('model_zoo_statistics.md', 'w') as f:
        f.write(modelzoo)


count_papers(model_index)


def generate_paper_page(collection):
    PAPERS_ROOT.mkdir(exist_ok=True)

    # Write a copy of README
    with open(MMSELFSUP_ROOT / collection.readme) as f:
        readme = f.read()
    folder = Path(collection.filepath).parent
    copy = PAPERS_ROOT / folder.with_suffix('.md').name

    def replace_link(matchobj):
        # Replace relative link to GitHub link.
        name = matchobj.group(1)
        link = matchobj.group(2)
        if not link.startswith('http'):
            assert (folder / link).exists(), \
                f'Link not found:\n{collection.readme}: {link}'
            rel_link = (folder / link).absolute().relative_to(MMSELFSUP_ROOT)
            link = GITHUB_PREFIX + str(rel_link)
        return f'[{name}]({link})'

    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, readme)

    with open(copy, 'w') as copy_file:
        copy_file.write(content)


for collection in model_index.collections:
    generate_paper_page(collection)
