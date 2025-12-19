import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--backbone', type=str, required=True)
args = parser.parse_args()

print(args.run_dir, args.method, args.backbone)

