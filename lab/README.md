# REMem Lab 🧪

A local, interactive Streamlit playground for the REMem memory framework. Tune
the knobs, build memory from documents, ask questions, and inspect exactly what
the system retrieved and reasoned over — so you can test REMem's promises
(memory construction → graph retrieval → multi-step QA) and see how each
variable changes behavior.

**This is local experimentation only.** It lives on the `lab/remem-playground`
branch and is intentionally kept out of any upstream PR. Do not add `streamlit`
to the repo's `pyproject.toml` / `requirements.txt`.

## Setup

```bash
# from the repo root, in the same env where `remem` is installed (-e .)
pip install -r lab/requirements-lab.txt
export OPENAI_API_KEY=sk-...        # required; online/OpenAI path only
python -m streamlit run lab/app.py
```

Runs on the **online / OpenAI path** (no vLLM) and defaults to OpenAI
embeddings (`text-embedding-3-small`), so nothing heavy is downloaded locally —
it works on macOS / non-CUDA machines.

## How to use it

1. **Input** — start with the bundled *Sample dataset* (3 docs, 1 gold
   question, indexes in seconds) or paste your own documents.
2. **Build memory** — runs OpenIE extraction + graph construction
   (≈ 2 LLM calls per chunk). Each distinct *build* configuration gets its own
   `outputs/lab/<hash>` working dir, so rebuilding the same config is instant
   (cached graph + embeddings + OpenIE results).
3. **Ask** — retrieves over the memory graph and answers. The **Retrieval / QA**
   sidebar knobs (`qa_top_k`, `agent_max_steps`, `damping`, …) apply on the next
   Ask **without** rebuilding. The **Build** knobs (`extract_method`,
   `graph_type`, `embedding_model_name`, chunking) require a rebuild.

The output panel shows the answer, the reasoning trace / agent session logs, the
retrieved passages with scores, the extracted facts used as graph seeds
(subject → predicate → object), the memory-graph stats, and EM / F1 / retrieval
recall when gold answers are available.

### Testing the promise
Switch `graph_type` between `dpr_only` (plain dense retrieval) and
`facts_and_sim_passage_node_unidirectional` (full graph memory), rebuild, and
re-ask the same question to compare whether the graph helps.

## Notes
- This branch does **not** include the held-back "surface OpenAI errors" fix
  (PR #7), so a failed API call may show up as an empty answer rather than a
  raised error. Check the reasoning-trace expander if an answer is blank.
- `working_dir`s accumulate under `outputs/lab/` (git-ignored). Delete that
  folder to reclaim space.
