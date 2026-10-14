"""
Stage 1 - run in the MAIN app environment (.venv), not the eval one.

Builds a separate Chroma index per variant from the same PDF, answers every
question in golden_set.json, and saves answers + retrieved contexts to
evaluation/answers.json.

The 3 original variants are unchanged (same settings as before):
  final      : full prompt, short-chunk filter ON, 4 context chunks, chunk 500/100
  baseline_a : simpler prompt (no document name, no multi-item instruction)
  baseline_b : full prompt, short-chunk filter OFF

New experiments (each changes ONE thing relative to `final`):
  final_k6       : 6 context chunks instead of 4
  final_partial  : prompt adds a rule for partially covered lists
  final_chunk800 : chunk_size=800, chunk_overlap=150

Usage:
  python evaluation\\generate_answers.py --pdf evaluation\\air2.pdf
      -> runs every variant (slow)
  python evaluation\\generate_answers.py --pdf evaluation\\air2.pdf --variants final_k6 final_partial final_chunk800
      -> runs only those and MERGES them into the existing answers.json
"""
import argparse
import json
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

FALLBACK = "I could not find this information in the document."

PROMPT_FULL = """You are a RAG assistant answering questions about an uploaded PDF document.

Answer the user's exact question using ONLY the context provided below.

Do not use outside knowledge.

Prioritize context that directly answers the question.
Ignore retrieved context that is only loosely related.

Start with a direct answer to the question.

If the question asks for multiple items, include all distinct items supported by the retrieved context and avoid duplicates.

If the answer cannot be found in the context, say:
"I could not find this information in the document."

Keep the answer clear and concise.

Document:
{active_document_name}

Question:
{question}

Context:
{context}

Answer:
"""

PROMPT_SIMPLE = """You are a RAG assistant answering questions about an uploaded PDF document.

Answer the user's exact question using ONLY the context provided below.

Do not use outside knowledge.

Prioritize context that directly answers the question.
Ignore retrieved context that is only loosely related.

Start with a direct answer to the question.

If the answer cannot be found in the context, say:
"I could not find this information in the document."

Keep the answer clear and concise.

Question:
{question}

Context:
{context}

Answer:
"""

# Same as PROMPT_FULL plus ONE extra rule (inserted after the multi-item rule).
PROMPT_PARTIAL = PROMPT_FULL.replace(
    "avoid duplicates.\n",
    "avoid duplicates.\n\n"
    "If the context covers only some of the requested items, list only the items that appear in the context, "
    "say that the remaining items are not in the provided context, and never add items that are not in the context.\n",
)
assert PROMPT_PARTIAL != PROMPT_FULL

BASE = {"prompt": PROMPT_FULL, "filter": True, "n_ctx": 4, "chunk": 500, "overlap": 100}

VARIANTS = {
    "final": dict(BASE),
    "baseline_a": {**BASE, "prompt": PROMPT_SIMPLE},
    "baseline_b": {**BASE, "filter": False},
    "final_k6": {**BASE, "n_ctx": 6},
    "final_partial": {**BASE, "prompt": PROMPT_PARTIAL},
    "final_chunk800": {**BASE, "chunk": 800, "overlap": 150},
}


def retrieve(store, question, cfg):
    candidates = store.max_marginal_relevance_search(
        question, k=8, fetch_k=15, lambda_mult=0.8
    )
    n = cfg["n_ctx"]
    if not cfg["filter"]:
        return candidates[:n]
    results = [d for d in candidates if len(d.page_content.strip()) >= 120][:n]
    if not results:
        results = candidates[:n]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--golden", default="evaluation/golden_set.json")
    ap.add_argument("--out", default="evaluation/answers.json")
    ap.add_argument("--chroma_dir", default="evaluation/chroma_bench")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {list(VARIANTS)}")

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    pdf_name = Path(args.pdf).name
    pages = PyPDFLoader(args.pdf).load()
    print(f"PDF pages: {len(pages)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    llm = ChatOllama(model="llama3.1:8b", temperature=0)

    out_path = Path(args.out)
    if out_path.exists():
        output = json.loads(out_path.read_text(encoding="utf-8"))
        output.setdefault("n_chunks_by_variant", {})
    else:
        output = {"pdf": pdf_name, "variants": {}, "n_chunks_by_variant": {}}

    chunk_cache = {}
    for name in args.variants:
        cfg = VARIANTS[name]
        key = (cfg["chunk"], cfg["overlap"])
        if key not in chunk_cache:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["chunk"], chunk_overlap=cfg["overlap"]
            )
            chunk_cache[key] = splitter.split_documents(pages)
        chunks = chunk_cache[key]
        output["n_chunks_by_variant"][name] = len(chunks)
        print(f"[{name}] chunks: {len(chunks)} (size {cfg['chunk']}/{cfg['overlap']}, context {cfg['n_ctx']})")

        folder = Path(args.chroma_dir) / name
        if folder.exists():
            shutil.rmtree(folder)
        store = Chroma.from_documents(chunks, embeddings, persist_directory=str(folder))

        rows = []
        for i, item in enumerate(golden, 1):
            docs = retrieve(store, item["question"], cfg)
            context = "\n\n".join(d.page_content for d in docs)
            prompt = cfg["prompt"].format(
                active_document_name=pdf_name,
                question=item["question"],
                context=context,
            )
            answer = llm.invoke(prompt).content.strip()
            rows.append(
                {
                    "question": item["question"],
                    "reference": item["reference"],
                    "answerable": item["answerable"],
                    "answer": answer,
                    "contexts": [d.page_content for d in docs],
                    "used_fallback": FALLBACK in answer,
                }
            )
            print(f"[{name}] {i}/{len(golden)} done")
        output["variants"][name] = rows  # merge/replace only this variant

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")

    print("\nUnanswerable questions (answerable=false):")
    for name in args.variants:
        rows = output["variants"][name]
        un = [r for r in rows if not r["answerable"]]
        fb = sum(r["used_fallback"] for r in un)
        print(f"  {name}: fallback={fb}/{len(un)}, other (possible hallucination)={len(un) - fb}")
        for r in un:
            print(f"    Q: {r['question']}\n    A: {r['answer']}\n")


if __name__ == "__main__":
    main()
