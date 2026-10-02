"""Opt-in real-model retrieval check for a freshly installed textgen environment."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from modules.sentence_rewrite.corpus import CorpusConfig, CorpusIndex
    from modules.sentence_rewrite.embeddings import EmbeddingConfig, LateInteractionEncoder
    from modules.sentence_rewrite.reranker import NuanceReranker
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = LateInteractionEncoder(EmbeddingConfig(
        device=args.device, local_files_only=args.offline, max_tokens=256,
    ))
    reranker = NuanceReranker(device=args.device, local_files_only=args.offline)
    index = CorpusIndex(args.output_dir / 'index')
    config = CorpusConfig(str(root / 'tests/fixtures/rewrite_reference_sentences.txt'), max_sentences=1)
    manifest = index.build(config, encoder, progress=lambda msg: print(msg, flush=True), force=True)
    hits = index.search('She reluctantly agreed to stay.', encoder, top_k=5,
                        reranker=reranker, rerank_pool=16,
                        progress=lambda msg: print(msg, flush=True))
    assert len(hits) == 5, hits
    assert all(math.isfinite(hit.score) and hit.nuance_score is not None for hit in hits)
    result = {
        'device': args.device, 'python': sys.version, 'torch_cuda': torch.version.cuda,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'sentence-transformers', 'gradio')},
        'manifest': manifest, 'hits': [asdict(hit) for hit in hits],
    }
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Real embedding, SQLite build, MaxSim search, and STS/NLI reranking passed.', flush=True)


if __name__ == '__main__':
    main()
