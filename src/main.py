# python "${ROOT_DIR}/src/main.py" --run-dir "${RUN_DIR}"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=str, required=True)
args = parser.parse_args()

print(args.run_dir)