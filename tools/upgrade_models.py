import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save-path', type=str, required=True, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author='OpenSelfSup')
    for key, value in ck.items():
        if key.startswith('head'):
            continue
        else:
            output_dict['state_dict'][key] = value
    torch.save(output_dict, args.save_path)


if __name__ == '__main__':
    main()
