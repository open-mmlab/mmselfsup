"""
SimCLR provides list files for semi-supervised benchmarks:
https://github.com/google-research/simclr/tree/master/imagenet_subsets/
This script convert the list files into the required format in OpenSelfSup.
"""
import argparse

parser = argparse.ArgumentParser(
    description='Convert ImageNet subset lists provided by simclr.')
parser.add_argument('input', help='Input list file.')
parser.add_argument('output', help='Output list file.')
args = parser.parse_args()

# create dict
with open("data/imagenet/meta/train_labeled.txt", 'r') as f:
    lines = f.readlines()
keys = [l.split('/')[0] for l in lines]
labels = [l.strip().split()[1] for l in lines]
mapping = {}
for k,l in zip(keys, labels):
    if k not in mapping:
        mapping[k] = l
    else:
        assert mapping[k] == l

# convert
with open(args.input, 'r') as f:
    lines = f.readlines()
fns = [l.strip() for l in lines]
sample_keys = [l.split('_')[0] for l in lines]
sample_labels = [mapping[k] for k in sample_keys]
output_lines = ["{}/{} {}\n".format(k, fn, l) for \
    k,fn,l in zip(sample_keys, fns, sample_labels)]
with open(args.output, 'w') as f:
    f.writelines(output_lines)
