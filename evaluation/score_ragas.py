"""
Stage 2 - run in the EVAL environment (.venv_eval).

Reads evaluation/answers.json (from generate_answers.py) and scores each
variant with RAGAS using a local Ollama judge.

Metrics:
  all questions        : faithfulness, response relevancy
  answerable=true only : context precision (with reference), factual correctness

Writes evaluation/results.csv (per question) and evaluation/summary.csv
(mean per metric per variant per run).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

try:
    from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision
except ImportError:  # older/newer ragas naming
    from ragas.metrics import ContextPrecision

try:
    from ragas.metrics import FactualCorrectness as Correctness
except ImportError:
    from ragas.metrics import AnswerCorrectness as Correctness

JUDGE_MODEL = "llama3.1:8b"
INPUT_COLS = {"user_input", "retrieved_contexts", "response", "reference"}


def to_samples(rows):
    return [
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["reference"],
        }
        for r in rows
    ]


def score(rows, metrics, llm, emb, run_config):
    ds = EvaluationDataset.from_list(to_samples(rows))
    res = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=run_config,
    )
    df = res.to_pandas()
    metric_cols = [c for c in df.columns if c not in INPUT_COLS]
    df["question"] = [r["question"] for r in rows]
    return df, metric_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="evaluation/answers.json")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--variants", nargs="*", default=None, help="score only these variants")
    ap.add_argument("--results", default="evaluation/results.csv")
    ap.add_argument("--summary", default="evaluation/summary.csv")
    args = ap.parse_args()

    data = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    print(f"Judge model: {JUDGE_MODEL} (local Ollama, same family as the app LLM)")

    llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text:latest"))
    run_config = RunConfig(timeout=900, max_workers=1)

    all_rows = []
    for run in range(1, args.runs + 1):
        for variant, rows in data["variants"].items():
            if args.variants and variant not in args.variants:
                continue
            print(f"\n=== run {run} | {variant} ===")
            answerable = [r for r in rows if r["answerable"]]

            df_all, _ = score(
                rows, [Faithfulness(), ResponseRelevancy()], llm, emb, run_config
            )
            df_ans, _ = score(
                answerable, [ContextPrecision(), Correctness()], llm, emb, run_config
            )

            keep_all = [c for c in df_all.columns if c not in INPUT_COLS]
            keep_ans = [c for c in df_ans.columns if c not in INPUT_COLS]
            merged = df_all[keep_all].merge(
                df_ans[keep_ans], on="question", how="left"
            )
            merged["answerable"] = [r["answerable"] for r in rows]
            merged["used_fallback"] = [r["used_fallback"] for r in rows]
            merged["variant"] = variant
            merged["run"] = run
            all_rows.append(merged)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(args.results, index=False, encoding="utf-8-sig")

    metric_cols = [
        c
        for c in results.columns
        if c not in {"question", "answerable", "used_fallback", "variant", "run"}
    ]
    summary = (
        results.groupby(["run", "variant"])[metric_cols].mean(numeric_only=True).round(4)
    )
    summary.to_csv(args.summary, encoding="utf-8-sig")
    print("\n=== SUMMARY (mean per metric) ===")
    print(summary.to_string())

    print("\n=== Unanswerable questions: fallback usage ===")
    un = results[(~results["answerable"]) & (results["run"] == 1)]
    print(un.groupby("variant")["used_fallback"].agg(["sum", "count"]).to_string())
    print(
        "\nLimitations: judge is a local 8B model (same family as the app LLM); "
        "metrics are noisy, so compare the two runs before drawing conclusions. "
        "Faithfulness/relevancy on fallback answers can be unreliable."
    )


if __name__ == "__main__":
    main()
