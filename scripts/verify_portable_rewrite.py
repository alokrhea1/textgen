"""Check a packaged Rewrite runtime without loading or downloading models."""
import argparse
import importlib
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--torch-backend", choices=("cpu", "cuda", "rocm"))
parser.add_argument("--cuda-version", help="Expected compiled CUDA version; no GPU is required.")
args = parser.parse_args()
if args.cuda_version and args.torch_backend != "cuda":
    parser.error("--cuda-version requires --torch-backend cuda")

# This script is shipped inside app/scripts in a portable archive.
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

for relative in ("modules/ui_sentence_rewrite.py", "modules/sentence_rewrite/embeddings.py",
                 "modules/sentence_rewrite/reranker.py", "modules/sentence_rewrite/corpus.py",
                 "modules/sentence_rewrite/engine.py", "LICENSE"):
    if not (root / relative).is_file():
        raise RuntimeError(f"Portable package is missing {relative}")

for name in ("torch", "transformers", "safetensors", "sentence_transformers",
             "modules.sentence_rewrite.embeddings", "modules.sentence_rewrite.reranker",
             "modules.sentence_rewrite.corpus", "modules.sentence_rewrite.engine"):
    importlib.import_module(name)

from sentence_transformers import SentenceTransformer
import torch

# CI runners need not have GPUs: verify the wheel's compiled runtime, not devices.
if args.torch_backend == "cuda":
    if not torch.version.cuda:
        raise RuntimeError("CUDA portable package contains a PyTorch build without CUDA support")
    if args.cuda_version and torch.version.cuda != args.cuda_version:
        raise RuntimeError(f"Expected CUDA {args.cuda_version} PyTorch, found {torch.version.cuda}")
elif args.torch_backend == "rocm" and not torch.version.hip:
    raise RuntimeError("ROCm portable package contains a PyTorch build without HIP support")
elif args.torch_backend == "cpu" and (torch.version.cuda or torch.version.hip):
    raise RuntimeError("CPU portable package unexpectedly contains CUDA/HIP PyTorch")

print("Portable Rewrite modules and dependencies are available (no models downloaded).")
