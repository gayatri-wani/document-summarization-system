import pandas as pd
from rouge_score import rouge_scorer

from .extractive import METHOD_LABELS, summarize

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge_eval(reference: str, candidate: str):
    s = _SCORER.score(reference, candidate)
    return {k: {"precision": v.precision, "recall": v.recall, "f1": v.fmeasure}
            for k, v in s.items()}


def compression_stats(original: str, summary: str):
    o, s = len(original.split()), len(summary.split())
    return {
        "original_words": o,
        "summary_words": s,
        "compression_%": round(100 * (1 - s / o), 1) if o else 0.0,
    }


def reading_time(words: int, wpm: int = 200) -> str:
    minutes = words / wpm
    return f"{minutes:.1f} min" if minutes >= 1 else f"{int(minutes * 60)} sec"


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def compare_methods(text, reference="", query="", ratio=0.2, num_sentences=None,
                    use_mmr=True, lam=0.7):
    """Run every extractive method; return a results table and a sentence-agreement matrix."""
    methods = ["textrank", "tfidf", "hybrid"] + (["bm25"] if query.strip() else [])
    rows, picked = [], {}
    for m in methods:
        r = summarize(text, m, num_sentences, ratio, query or None, use_mmr, lam)
        label = METHOD_LABELS[m]
        row = {
            "Method": label,
            "Sentences": len(r["selected"]),
            "Words": len(r["summary"].split()),
            "Time (ms)": round(r["elapsed_ms"], 1),
        }
        if reference.strip():
            s = rouge_eval(reference, r["summary"])
            row["ROUGE-1"] = round(s["rouge1"]["f1"], 3)
            row["ROUGE-2"] = round(s["rouge2"]["f1"], 3)
            row["ROUGE-L"] = round(s["rougeL"]["f1"], 3)
        rows.append(row)
        picked[label] = set(r["selected"])

    names = list(picked)
    agreement = pd.DataFrame(
        [[round(_jaccard(picked[a], picked[b]), 2) for b in names] for a in names],
        index=names, columns=names,
    )
    return pd.DataFrame(rows), agreement