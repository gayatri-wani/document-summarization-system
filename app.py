import html
import os
import time

import pandas as pd
import streamlit as st

from summarizer.evaluate import compare_methods, compression_stats, reading_time, rouge_eval
from summarizer.export import summary_to_docx
from summarizer.extractive import METHOD_LABELS, summarize
from summarizer.loader import load_text
from summarizer.search import DocumentIndex
from summarizer.storage import clear_history, delete_history, init_db, list_history, save_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

st.set_page_config(page_title="DocSummarizer", page_icon="📄", layout="wide")
init_db()

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #db2777 100%);
       color: white; padding: 28px 34px; border-radius: 18px; margin-bottom: 20px;
       box-shadow: 0 10px 30px rgba(79,70,229,.30);}
.hero h1 {color: white; margin: 0; font-size: 2.1rem; padding: 0;}
.hero p {margin: 8px 0 0; opacity: .92; font-size: 1.02rem;}
.summary-card {background: white; border-left: 6px solid #6366f1; padding: 20px 24px;
       border-radius: 12px; box-shadow: 0 2px 12px rgba(15,23,42,.08);
       line-height: 1.75; font-size: 1.04rem;}
.doc-view {background: white; padding: 20px 24px; border-radius: 12px; line-height: 1.95;
       box-shadow: 0 2px 12px rgba(15,23,42,.06); font-size: .98rem;}
.sel {background: #fef08a; border-radius: 5px; padding: 2px 4px; font-weight: 600;}
.badge {background: #6366f1; color: white; border-radius: 8px; font-size: .65rem;
       padding: 1px 5px; margin-left: 3px;}
.chip {display: inline-block; background: #e0e7ff; color: #3730a3; padding: 4px 13px;
       margin: 3px 5px 3px 0; border-radius: 999px; font-size: .85rem; font-weight: 600;}
.result-card {background: white; padding: 16px 20px; border-radius: 12px; margin-bottom: 10px;
       border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(15,23,42,.05);}
.result-card .score {color: #6366f1; font-weight: 700;}
[data-testid="stMetric"] {background: white; padding: 14px 16px; border-radius: 12px;
       box-shadow: 0 2px 10px rgba(15,23,42,.06);}
</style>
<div class="hero">
  <h1>📄 DocSummarizer</h1>
  <p>Information Retrieval powered summarization: TF-IDF · TextRank · BM25 · MMR</p>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ cached backend calls
@st.cache_data(show_spinner=False, max_entries=64)
def cached_extractive(text, method, num_sentences, ratio, query, use_mmr, lam):
    return summarize(text, method, num_sentences, ratio, query, use_mmr, lam)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_abstractive(text):
    from summarizer.abstractive import abstractive_summary
    return abstractive_summary(text)


def chips_html(words):
    return "".join(f'<span class="chip">{html.escape(w)}</span>' for w in words)


def join_docs(texts):
    fixed = [t.strip() if t.strip().endswith((".", "!", "?")) else t.strip() + "." for t in texts]
    return " ".join(fixed)


# ------------------------------------------------------------------ sidebar
LABEL_TO_KEY = {v: k for k, v in METHOD_LABELS.items()}
LABEL_TO_KEY["Abstractive (DistilBART)"] = "abstractive"

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    method_label = st.selectbox("Summarization method", list(LABEL_TO_KEY.keys()))
    method_key = LABEL_TO_KEY[method_label]

    size_mode = st.radio("Summary length by", ["Ratio", "Sentences"], horizontal=True)
    ratio, num_sent = 0.2, None
    if size_mode == "Ratio":
        ratio = st.slider("Share of original", 0.05, 0.6, 0.2, 0.05)
    else:
        num_sent = int(st.number_input("Number of sentences", 1, 50, 5))

    use_mmr = st.toggle("Reduce redundancy (MMR)", True)
    lam = st.slider("MMR λ  (relevance ↔ diversity)", 0.3, 1.0, 0.7, 0.05, disabled=not use_mmr)

    st.divider()
    query = st.text_input(
        "Focus query (optional)",
        help="Required for BM25. For Hybrid it boosts sentences relevant to the query.",
        placeholder="e.g. machine learning applications",
    ).strip()
    st.caption("Tip: Abstractive needs `torch` + `transformers` installed.")

tab_sum, tab_cmp, tab_search, tab_hist = st.tabs(
    ["✨ Summarize", "📊 Compare methods", "🔎 Search & summarize", "🕘 History"]
)

# ================================================================== TAB 1: SUMMARIZE
with tab_sum:
    source = st.radio("Input", ["Upload file", "Paste text", "Sample"],
                      horizontal=True, label_visibility="collapsed")
    text, title = "", "Untitled"

    if source == "Upload file":
        up = st.file_uploader("Upload TXT, PDF or DOCX", type=["txt", "pdf", "docx"])
        if up:
            try:
                text, title = load_text(up), up.name
            except Exception as e:
                st.error(f"Could not read file: {e}")
    elif source == "Paste text":
        text = st.text_area("Paste your document", height=260,
                            placeholder="Paste at least a few sentences...")
        title = "Pasted text"
    else:
        sample_path = os.path.join(BASE_DIR, "samples", "sample.txt")
        if os.path.exists(sample_path):
            with open(sample_path, encoding="utf-8", errors="ignore") as fh:
                text, title = fh.read(), "sample.txt"
            st.caption(f"Loaded samples/sample.txt ({len(text.split()):,} words)")
        else:
            st.info("Add a file at samples/sample.txt to use this option.")

    text = text.strip()
    st.session_state["text"] = text
    if text:
        st.caption(f"📏 {len(text.split()):,} words · about {reading_time(len(text.split()))} to read")

    with st.expander("Optional: reference summary for ROUGE evaluation"):
        st.text_area("Human-written reference summary", key="ref_text", height=110)

    if st.button("✨ Generate summary", type="primary"):
        if len(text.split()) < 30:
            st.warning("Please provide a longer document (at least ~30 words).")
        else:
            with st.spinner("Analysing document..."):
                t0 = time.perf_counter()
                try:
                    if method_key == "abstractive":
                        pre = cached_extractive(text, "hybrid", None, 0.4, query or None, use_mmr, lam)
                        summary = cached_abstractive(pre["summary"] or text)
                        res = {"summary": summary, "sentences": pre["sentences"], "selected": [],
                               "scores": pre["scores"], "keywords": pre["keywords"], "abstractive": True}
                    else:
                        res = dict(cached_extractive(text, method_key, num_sent, ratio,
                                                     query or None, use_mmr, lam))
                        res["abstractive"] = False
                except ImportError:
                    st.error("Abstractive mode needs: pip install torch transformers")
                    st.stop()
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            stats = compression_stats(text, res["summary"])
            res.update(stats)
            res.update({"title": title, "method_label": method_label, "query": query,
                        "total_ms": (time.perf_counter() - t0) * 1000})
            st.session_state["res"] = res
            save_summary(title, method_label, query, stats["original_words"],
                         stats["summary_words"], res["summary"])

    res = st.session_state.get("res")
    if res:
        if not res["summary"]:
            st.warning("No summary could be generated for this text.")
        else:
            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Original words", f"{res['original_words']:,}")
            m2.metric("Summary words", f"{res['summary_words']:,}")
            m3.metric("Compression", f"{res['compression_%']}%")
            m4.metric("Reading time", reading_time(res["summary_words"]),
                      delta=f"saves {reading_time(res['original_words'] - res['summary_words'])}",
                      delta_color="off")
            m5.metric("Processing", f"{res['total_ms']:.0f} ms")

            st.markdown(f"#### 📝 Summary  ·  _{res['method_label']}_")
            st.markdown(f'<div class="summary-card">{html.escape(res["summary"])}</div>',
                        unsafe_allow_html=True)

            d1, d2, _ = st.columns([1, 1, 3])
            d1.download_button("⬇️ Download .txt", res["summary"], file_name="summary.txt")
            d2.download_button(
                "⬇️ Download .docx",
                summary_to_docx(res["title"], res["summary"], res["keywords"],
                                f"Method: {res['method_label']}"),
                file_name="summary.docx", mime=DOCX_MIME)

            if res["keywords"]:
                st.markdown("**🔑 Top keywords**")
                st.markdown(chips_html(res["keywords"]), unsafe_allow_html=True)

            t_doc, t_scores, t_rouge = st.tabs(
                ["Highlighted document", "Sentence scores", "ROUGE evaluation"])

            with t_doc:
                if res["abstractive"]:
                    st.info("Abstractive summaries generate new text, so there are no highlighted sentences.")
                else:
                    sel = {idx: rank + 1 for rank, idx in enumerate(res["selected"])}
                    parts = []
                    for i, s in enumerate(res["sentences"]):
                        safe = html.escape(s)
                        if i in sel:
                            parts.append(f'<span class="sel" title="score {res["scores"][i]:.2f}">'
                                         f'{safe}<span class="badge">{sel[i]}</span></span>')
                        else:
                            parts.append(safe)
                    st.markdown(f'<div class="doc-view">{" ".join(parts)}</div>', unsafe_allow_html=True)

            with t_scores:
                if res["scores"]:
                    chosen = set(res["selected"])
                    df = pd.DataFrame({
                        "#": range(1, len(res["scores"]) + 1),
                        "score": [round(x, 3) for x in res["scores"]],
                        "selected": ["✅" if i in chosen else "" for i in range(len(res["scores"]))],
                        "sentence": res["sentences"],
                    })
                    st.bar_chart(df.set_index("#")["score"])
                    st.dataframe(df.sort_values("score", ascending=False), hide_index=True)

            with t_rouge:
                ref = st.session_state.get("ref_text", "").strip()
                if ref:
                    r = rouge_eval(ref, res["summary"])
                    rdf = pd.DataFrame(r).T.round(3)
                    st.dataframe(rdf)
                    st.bar_chart(rdf["f1"])
                else:
                    st.info("Paste a reference summary in the expander above to see ROUGE scores.")

# ================================================================== TAB 2: COMPARE
with tab_cmp:
    st.markdown("Run **all extractive methods** on the document from the Summarize tab and compare them.")
    cmp_text = st.session_state.get("text", "")
    if not cmp_text:
        st.info("Load a document in the Summarize tab first.")
    elif st.button("📊 Run comparison"):
        with st.spinner("Comparing methods..."):
            table, agreement = compare_methods(
                cmp_text, st.session_state.get("ref_text", ""), query,
                ratio, num_sent, use_mmr, lam)
        st.session_state["cmp"] = (table, agreement)

    if "cmp" in st.session_state:
        table, agreement = st.session_state["cmp"]
        st.markdown("#### Results")
        st.dataframe(table, hide_index=True)
        if "ROUGE-L" in table.columns:
            st.bar_chart(table.set_index("Method")[["ROUGE-1", "ROUGE-2", "ROUGE-L"]])
        else:
            st.caption("Add a reference summary (Summarize tab) to get ROUGE scores.")
        st.markdown("#### Sentence agreement between methods (Jaccard)")
        st.caption("1.0 means two methods picked exactly the same sentences.")
        st.dataframe(agreement)

# ================================================================== TAB 3: SEARCH
with tab_search:
    st.markdown("Upload **several documents**, search them with **BM25**, then summarize the best matches.")
    files = st.file_uploader("Upload documents", type=["txt", "pdf", "docx"],
                             accept_multiple_files=True, key="multi")
    search_q = st.text_input("Search query", placeholder="e.g. neural networks in healthcare", key="sq")
    top_k = st.slider("Documents to retrieve", 1, 10, 3)

    if st.button("🔎 Search & summarize"):
        if not files or not search_q.strip():
            st.warning("Upload documents and enter a query.")
        else:
            docs = {}
            for f in files:
                try:
                    docs[f.name] = load_text(f)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if docs:
                hits = DocumentIndex(docs).search(search_q, top_k)
                if not hits:
                    st.info("No documents matched that query.")
                else:
                    st.markdown(f"#### Top {len(hits)} result(s)")
                    for rank, h in enumerate(hits, 1):
                        st.markdown(
                            f'<div class="result-card"><b>{rank}. {html.escape(h["name"])}</b> '
                            f'&nbsp;<span class="score">BM25 {h["score"]:.2f}</span><br>'
                            f'<i>{html.escape(h["snippet"])}</i></div>',
                            unsafe_allow_html=True)
                        with st.expander(f"Query-focused summary of {h['name']}"):
                            try:
                                r = summarize(h["text"], "bm25", 3, 0.2, search_q, True, 0.7)
                                st.write(r["summary"] or "Not enough text to summarize.")
                            except Exception as e:
                                st.write(str(e))
                    st.markdown("#### 🧩 Combined summary of all retrieved documents")
                    combined = join_docs([h["text"] for h in hits])
                    r = summarize(combined, "hybrid", 5, 0.2, search_q, True, 0.7)
                    st.markdown(f'<div class="summary-card">{html.escape(r["summary"])}</div>',
                                unsafe_allow_html=True)

# ================================================================== TAB 4: HISTORY
with tab_hist:
    rows = list_history()
    if not rows:
        st.info("No summaries yet. Generate one and it will appear here.")
    else:
        top1, top2 = st.columns([4, 1])
        top1.markdown(f"**{len(rows)} saved summaries** (stored in a local SQLite database)")
        if top2.button("🗑️ Clear all"):
            clear_history()
            st.rerun()
        for row in rows:
            label = f"{row['created_at']} · {row['title']} · {row['method']}"
            with st.expander(label):
                if row["query"]:
                    st.caption(f"Query: {row['query']}")
                st.write(row["summary"])
                st.caption(f"{row['original_words']:,} → {row['summary_words']:,} words")
                if st.button("Delete", key=f"del_{row['id']}"):
                    delete_history(row["id"])
                    st.rerun()