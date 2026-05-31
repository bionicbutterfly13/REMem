# Fork changes

This is a modified fork of [intuit-ai-research/REMem](https://github.com/intuit-ai-research/REMem),
the official implementation of *REMem: Reasoning with Episodic Memory in Language Agents* (ICLR 2026).
Licensed under Apache License 2.0; see `LICENSE`.

Modifications from upstream (per Apache 2.0 §4(b), stating significant changes):

## macOS / non-GPU support
- `pyproject.toml` and `requirements.txt`: removed the `vllm==0.8.5post1` pin
  (CUDA/Linux-only, no macOS build) and unpinned `torch` (the pinned `2.6.0` has no
  Intel-Mac wheel; pip now resolves to a platform-compatible torch such as 2.2.2).
  Both files are updated so installs via `pyproject.toml` *or* `setup.py` succeed on macOS.

## Dataset path resolution
- `main.py`: datasets ship in two layouts — flat (`reproduce/dataset/<name>_corpus.json`)
  and nested (`reproduce/dataset/<name>/<name>_corpus.json`). The original code only
  resolved the flat layout, so `--dataset musique` (and other nested datasets) failed with
  `FileNotFoundError`. Added `_resolve_dataset_path` to prefer whichever layout exists.

These changes only affect installation. The vLLM offline code paths
(`*_vllm_offline.py`) are unchanged and remain available where a compatible
environment provides vLLM; they are simply no longer a hard install dependency.

To run on macOS, use the online (API) backend, e.g.:

```bash
export OPENAI_API_KEY=sk-...
python main.py --dataset musique \
  --llm_base_url https://api.openai.com/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small \
  --llm_infer_mode online
```
