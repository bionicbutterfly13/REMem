"""Minimal end-to-end smoke test for the online/OpenAI (no-CUDA) path.

Reuses the cached `sample` build (gpt-5-nano + text-embedding-3-small) so
extraction is a pure cache hit -- proves: install OK -> load graph -> retrieve
-> answer, live, without spending fresh extraction calls.

Run: source .venv/bin/activate && python lab/smoke_test.py
"""
import json

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig

config = BaseConfig(
    llm_base_url="https://api.openai.com/v1/",
    llm_name="gpt-5-nano",                            # matches cached openie/llm cache
    dataset="sample",                                  # save_dir -> outputs/sample
    embedding_model_name="text-embedding-3-small",     # matches cached embedding store
    llm_infer_mode="online",
    rerank_dspy_file_path=None,
    extract_method="openie",
    graph_type="facts_and_sim_passage_node_unidirectional",
    embedding_batch_size=8,
    force_index_from_scratch=False,                    # reuse cached graph + openie
    do_eval_retrieval=False,
    do_eval_qa=False,
)

work = "outputs/sample/sample_gpt-5-nano_text-embedding-3-small"
rag = ReMem(global_config=config, working_dir=work)

corpus = json.load(open("reproduce/dataset/sample_corpus.json"))
docs = [item["text"] for item in corpus]
rag.index(docs)  # cache hit -> loads existing graph, no fresh extraction

sample = json.load(open("reproduce/dataset/sample.json"))
queries = [item["question"] for item in sample]
gold = [item["answer"] for item in sample]

result = rag.rag_for_qa(queries)
solutions = result[0]

print("\n===== SMOKE TEST RESULTS =====")
for s, g in zip(solutions, gold):
    print(f"Q:    {s.question}")
    print(f"A:    {s.answer}")
    print(f"Gold: {g}\n")
print("graph info:", rag.get_graph_info())
