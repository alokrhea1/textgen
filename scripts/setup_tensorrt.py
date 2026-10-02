#!/usr/bin/env python3
"""Install the optional TensorRT backend without changing webui dependencies."""
import argparse
import ctypes.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', help='Python 3.12 executable (auto-detected by default)')
    parser.add_argument('--runtime-dir', type=Path, default=root / 'user_data/tensorrt_runtime')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('TensorRT-LLM runtime setup requires Linux')
    python = args.python or (sys.executable if sys.version_info[:2] == (3, 12) else shutil.which('python3.12'))
    if not python:
        parser.error('Install Python 3.12, then rerun with --python /path/to/python3.12')
    version = subprocess.check_output([python, '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
    if version != '3.12':
        parser.error(f'This pinned runtime requires Python 3.12; selected Python is {version}')
    if not ctypes.util.find_library('mpi'):
        parser.error('OpenMPI is missing. On Ubuntu/Debian install libopenmpi-dev, then rerun this script.')
    destination = args.runtime_dir.resolve()
    print(f'Installing TensorRT-LLM 1.0.0 into {destination}', flush=True)
    subprocess.run([python, '-m', 'venv', str(destination)], check=True)
    runtime = str(destination / 'bin/python')
    subprocess.run([runtime, '-m', 'pip', 'install', 'pip', 'setuptools<80', 'wheel<=0.45.1'], check=True)
    with tempfile.TemporaryDirectory(prefix='tensorrt-constraints-') as temp:
        constraints = Path(temp) / 'constraints.txt'
        constraints.write_text('torch==2.7.1\ntorchvision==0.22.1\ntransformers==4.53.1\nnumpy==1.26.4\n')
        subprocess.run([runtime, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1',
                        '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
        subprocess.run([runtime, '-m', 'pip', 'install', 'tensorrt-llm==1.0.0',
                        '--extra-index-url', 'https://pypi.nvidia.com', '-c', str(constraints)], check=True)
        subprocess.run([runtime, '-m', 'pip', 'check'], check=True)
    print(f'Runtime ready. Start webui with --loader TensorRT-LLM --ctx-size 8192 --tensorrt-llm-python {runtime}', flush=True)


if __name__ == '__main__':
    main()
