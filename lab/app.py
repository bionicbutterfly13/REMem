"""
REMem Lab — an interactive Streamlit playground for the REMem memory framework.

Goal: let you tune REMem's variables, ingest documents, *build memory*, ask
questions, and see the evidence the system actually used — so you can test the
software's promises (memory construction -> graph retrieval -> multi-step QA)
and watch how each knob changes behavior.

Runs on the ONLINE / OpenAI path only (no vLLM, works on macOS). Defaults to
OpenAI embeddings so nothing heavy is downloaded locally.

Run:
    pip install -r lab/requirements-lab.txt
    export OPENAI_API_KEY=...        # already set in this shell if you used the project env
    python -m streamlit run lab/app.py

This file lives only on the local `lab/remem-playground` branch and is never
part of an upstream PR.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import streamlit as st

# --- make the repo importable regardless of where streamlit is launched from ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from remem.remem import ReMem  # noqa: E402
from remem.utils.config_utils import BaseConfig  # noqa: E402

# Reuse main.py's gold-doc/answer extraction so the lab matches the real entry point.
try:
    from main import get_gold_docs, get_gold_answers  # noqa: E402
except Exception:  # pragma: no cover - fallback if import shape changes
    get_gold_docs = None
    get_gold_answers = None

# --------------------------------------------------------------------------- #
# Constants — the online-relevant subset of BaseConfig knobs.
# --------------------------------------------------------------------------- #
OPENAI_BASE_URL = "https://api.openai.com/v1"

LLM_CHOICES = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
EMBEDDING_CHOICES = ["text-embedding-3-small", "text-embedding-3-large"]
EXTRACT_METHODS = ["openie", "episodic", "episodic_gist", "temporal"]
CHUNK_FUNCS = ["by_token", "by_word", "by_message", "by_session", "none"]
GRAPH_TYPES = [
    "facts_and_sim_passage_node_unidirectional",
    "facts_and_sim",
    "dpr_only",
]

SAMPLE_CORPUS = os.path.join(REPO_ROOT, "reproduce", "dataset", "sample_corpus.json")
SAMPLE_QA = os.path.join(REPO_ROOT, "reproduce", "dataset", "sample.json")
LAB_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "lab")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_sample_dataset():
    """Return (docs, queries, gold_docs, gold_answers) for the bundled sample set."""
    with open(SAMPLE_CORPUS, "r") as f:
        corpus = json.load(f)
    with open(SAMPLE_QA, "r") as f:
        samples = json.load(f)

    docs = [f"{d['title']}\n{d['text']}" for d in corpus]
    queries = [s["question"] for s in samples]

    if get_gold_docs and get_gold_answers:
        gold_docs = get_gold_docs(samples, "sample")
        gold_answers = [list(a) for a in get_gold_answers(samples)]
    else:  # inline fallback mirroring main.py's "paragraphs"/"answer" branch
        gold_docs, gold_answers = [], []
        for s in samples:
            gd = [
                f"{p['title']}\n{p.get('text', p.get('paragraph_text', ''))}"
                for p in s.get("paragraphs", [])
                if p.get("is_supporting", True)
            ]
            gold_docs.append(list(set(gd)))
            ans = s.get("answer")
            gold_answers.append(ans if isinstance(ans, list) else [ans])

    return docs, queries, gold_docs, gold_answers


# --------------------------------------------------------------------------- #
# Config + memory build
# --------------------------------------------------------------------------- #
def build_identity(build_knobs: dict, docs: list[str]) -> str:
    """Hash of build-relevant knobs + the documents. Retrieval/QA knobs are
    intentionally excluded so tuning them does not force a re-index."""
    payload = json.dumps({"build": build_knobs, "docs": docs}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def make_config(build_knobs: dict, qa_knobs: dict, n_docs: int) -> BaseConfig:
    max_tok = build_knobs["chunk_max_tokens"]
    return BaseConfig(
        llm_base_url=OPENAI_BASE_URL,
        llm_name=build_knobs["build_llm"],
        dataset="lab",
        embedding_model_name=build_knobs["embedding_model_name"],
        llm_infer_mode="online",
        rerank_dspy_file_path=None,          # no DSPy reranker — keep the demo self-contained
        extract_method=build_knobs["extract_method"],
        preprocess_chunk_func=build_knobs["chunk_func"],
        preprocess_chunk_max_token_size=(max_tok if max_tok and max_tok > 0 else None),
        graph_type=build_knobs["graph_type"],
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=n_docs,
        # retrieval / QA knobs (read at query time; updated live before each Ask)
        retrieval_top_k=qa_knobs["retrieval_top_k"],
        linking_top_k=qa_knobs["linking_top_k"],
        qa_top_k=qa_knobs["qa_top_k"],
        damping=qa_knobs["damping"],
        agent_fixed_tools=qa_knobs["agent_fixed_tools"],
        agent_max_steps=qa_knobs["agent_max_steps"],
        temperature=qa_knobs["temperature"],
        do_eval_retrieval=False,
        do_eval_qa=False,
        force_index_from_scratch=build_knobs["force_rebuild"],
    )


def apply_qa_knobs(remem: ReMem, qa_knobs: dict) -> None:
    """Push current retrieval/QA slider values onto the live config so they take
    effect on the next Ask without rebuilding the graph."""
    cfg = remem.global_config
    cfg.retrieval_top_k = qa_knobs["retrieval_top_k"]
    cfg.linking_top_k = qa_knobs["linking_top_k"]
    cfg.qa_top_k = qa_knobs["qa_top_k"]
    cfg.damping = qa_knobs["damping"]
    cfg.agent_fixed_tools = qa_knobs["agent_fixed_tools"]
    cfg.agent_max_steps = qa_knobs["agent_max_steps"]
    cfg.temperature = qa_knobs["temperature"]
    if qa_knobs["qa_llm"] != qa_knobs.get("build_llm"):
        cfg.qa_llm_label = qa_knobs["qa_llm"]


# --------------------------------------------------------------------------- #
# Output rendering
# --------------------------------------------------------------------------- #
def render_solution(sol, overall_ret: dict, overall_qa: dict):
    st.subheader("Answer")
    st.success(sol.answer if sol.answer else "_(empty — check the reasoning trace / API errors below)_")

    cols = st.columns(3)
    metrics = sol.metrics or {}
    em = metrics.get("ExactMatch", metrics.get("qa_em"))
    f1 = metrics.get("F1", metrics.get("qa_f1"))
    recall = None
    for k, v in (overall_ret or {}).items():
        if "recall" in k.lower():
            recall = v
            break
    cols[0].metric("Exact Match", f"{em:.2f}" if isinstance(em, (int, float)) else "—")
    cols[1].metric("F1", f"{f1:.2f}" if isinstance(f1, (int, float)) else "—")
    cols[2].metric("Retrieval recall", f"{recall:.2f}" if isinstance(recall, (int, float)) else "—")

    with st.expander("🧠 Reasoning trace", expanded=bool(sol.qa_rationale)):
        st.write(sol.qa_rationale or "_(no rationale returned)_")
        if sol.agent_session_logs:
            st.caption("Agent session logs (multi-step reasoning)")
            st.json(_safe(sol.agent_session_logs))

    with st.expander("📚 Retrieved passages (the memory the answer used)", expanded=True):
        docs = sol.docs or []
        scores = sol.doc_scores.tolist() if hasattr(sol.doc_scores, "tolist") else (sol.doc_scores or [])
        if not docs:
            st.info("No passages retrieved.")
        for i, doc in enumerate(docs[: max(1, len(docs))][:10]):
            score = f"{scores[i]:.4f}" if i < len(scores) else "—"
            st.markdown(f"**#{i + 1}** · score `{score}`")
            st.write(doc[:1200] + ("…" if len(doc) > 1200 else ""))
            st.divider()

    with st.expander("🔗 Extracted facts used as graph seeds (subject → predicate → object)"):
        seeds = sol.graph_seeds or []
        if not seeds:
            st.info("No graph seeds (dense-retrieval fallback, or graph_type=dpr_only).")
        for seed in seeds[:25]:
            if isinstance(seed, (list, tuple)) and len(seed) == 3:
                st.markdown(f"- **{seed[0]}** → _{seed[1]}_ → **{seed[2]}**")
            else:
                st.markdown(f"- `{seed}`")

    with st.expander("🔍 Raw QuerySolution (safe JSON)"):
        st.json(_safe(sol.to_dict()))


def _safe(obj):
    """Best-effort make something json-renderable for st.json."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="REMem Lab", page_icon="🧪", layout="wide")
st.title("🧪 REMem Lab")
st.caption("Tune the knobs → build memory → ask → inspect the evidence. Online/OpenAI path only.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set in this environment. Export it before building memory.")

# ---------------- Sidebar: knobs ----------------
with st.sidebar:
    st.header("⚙️ Build knobs")
    st.caption("Changing these rebuilds the memory graph.")
    extract_method = st.selectbox("extract_method", EXTRACT_METHODS, index=0)
    embedding_model_name = st.selectbox("embedding_model_name", EMBEDDING_CHOICES, index=0)
    build_llm = st.selectbox("extraction llm_name", LLM_CHOICES, index=0)
    graph_type = st.selectbox("graph_type", GRAPH_TYPES, index=0)
    chunk_func = st.selectbox("preprocess_chunk_func", CHUNK_FUNCS, index=0)
    chunk_max_tokens = st.slider("chunk_max_token_size (0 = whole doc)", 0, 2048, 0, step=64)

    st.header("🎛️ Retrieval / QA knobs")
    st.caption("Changing these does NOT rebuild — just re-Ask.")
    retrieval_top_k = st.slider("retrieval_top_k", 1, 200, 50)
    linking_top_k = st.slider("linking_top_k", 1, 20, 5)
    qa_top_k = st.slider("qa_top_k", 1, 20, 5)
    damping = st.slider("damping (PPR)", 0.0, 1.0, 0.5, step=0.05)
    agent_fixed_tools = st.checkbox("agent_fixed_tools", value=False)
    agent_max_steps = st.slider("agent_max_steps", 1, 10, 3)
    qa_llm = st.selectbox("QA llm_name", LLM_CHOICES, index=0, key="qa_llm")
    temperature = st.slider("temperature", 0.0, 2.0, 0.0, step=0.1)

    with st.expander("Advanced"):
        force_rebuild = st.checkbox("force_index_from_scratch", value=False)

build_knobs = {
    "extract_method": extract_method,
    "embedding_model_name": embedding_model_name,
    "build_llm": build_llm,
    "graph_type": graph_type,
    "chunk_func": chunk_func,
    "chunk_max_tokens": chunk_max_tokens,
    "force_rebuild": force_rebuild,
}
qa_knobs = {
    "retrieval_top_k": retrieval_top_k,
    "linking_top_k": linking_top_k,
    "qa_top_k": qa_top_k,
    "damping": damping,
    "agent_fixed_tools": agent_fixed_tools,
    "agent_max_steps": agent_max_steps,
    "qa_llm": qa_llm,
    "build_llm": build_llm,
    "temperature": temperature,
}

# ---------------- Input ----------------
st.header("1 · Input")
source = st.radio("Documents to remember", ["Sample dataset (instant)", "Custom"], horizontal=True)

if source.startswith("Sample"):
    docs, queries, gold_docs, gold_answers = load_sample_dataset()
    st.info(f"Loaded **{len(docs)}** documents and **{len(queries)}** gold question(s).")
    default_question = queries[0] if queries else ""
    has_gold = True
else:
    raw = st.text_area(
        "Paste documents (separate multiple docs with a blank line)",
        height=180,
        placeholder="Alice went to the store and bought milk.\n\nBob called Alice yesterday.",
    )
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
    gold_docs = gold_answers = None
    default_question = ""
    has_gold = False
    st.info(f"{len(docs)} document(s) ready." if docs else "Paste at least one document.")

# ---------------- Build ----------------
st.header("2 · Build memory")
cur_hash = build_identity(build_knobs, docs) if docs else None
state = st.session_state

stale = state.get("build_hash") and state.get("build_hash") != cur_hash
if stale:
    st.warning("Build knobs or documents changed since the last build — rebuild to apply.")

if st.button("🧱 Build memory", type="primary", disabled=not docs):
    working_dir = os.path.join(LAB_OUTPUT_ROOT, cur_hash)
    cfg = make_config(build_knobs, qa_knobs, len(docs))
    with st.spinner(f"Building memory ({len(docs)} docs · OpenIE ≈ 2 LLM calls/chunk)…"):
        t0 = time.time()
        remem = ReMem(global_config=cfg, working_dir=working_dir)
        remem.index(docs)
        dt = time.time() - t0
    state["remem"] = remem
    state["build_hash"] = cur_hash
    state["working_dir"] = working_dir
    st.success(f"Memory built in {dt:.1f}s · working_dir `{os.path.relpath(working_dir, REPO_ROOT)}`")

if state.get("remem") is not None and state.get("build_hash") == cur_hash:
    try:
        ginfo = state["remem"].get_graph_info()
        gc = st.columns(len(ginfo) or 1)
        for i, (k, v) in enumerate(ginfo.items()):
            gc[i % len(gc)].metric(k, v)
    except Exception as e:  # get_graph_info shape can vary by graph_type
        st.caption(f"(graph info unavailable: {e})")

# ---------------- Ask ----------------
st.header("3 · Ask")
question = st.text_input("Question", value=default_question)
ready = state.get("remem") is not None and state.get("build_hash") == cur_hash and bool(question)

if st.button("💬 Ask", type="primary", disabled=not ready):
    remem = state["remem"]
    apply_qa_knobs(remem, qa_knobs)

    gd = ga = None
    if has_gold and question in queries:
        idx = queries.index(question)
        gd = [gold_docs[idx]]
        ga = [gold_answers[idx]]

    with st.spinner("Retrieving over the memory graph and answering…"):
        result = remem.rag_for_qa(
            queries=[question],
            gold_docs=gd,
            gold_answers=ga,
            metrics=("qa_em", "qa_f1", "retrieval_recall"),
        )
    solutions = result[0]
    overall_ret = result[3] if len(result) >= 5 else {}
    overall_qa = result[4] if len(result) >= 5 else {}
    render_solution(solutions[0], overall_ret, overall_qa)
elif not ready and question and state.get("build_hash") != cur_hash:
    st.info("Build memory first (knobs changed).")
