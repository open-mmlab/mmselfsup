import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file):
    tmp_file = in_file + ".tmp"
    subprocess.Popen(['cp', in_file, tmp_file])
    sha = subprocess.check_output(['sha256sum', tmp_file]).decode()
    out_file = in_file
    if out_file.endswith('.pth'):
        out_file = out_file[:-4]
    final_file = out_file + f'-{sha[:8]}.pth'
    assert final_file != in_file, \
        "The output filename is the same as the input file."
    print("Output file: {}".format(final_file))
    subprocess.Popen(['mv', tmp_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file)


if __name__ == '__main__':
    main()
