_pipe = None
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline  # lazy import so extractive works without it
        _pipe = pipeline("summarization", model=MODEL_NAME)
    return _pipe


def abstractive_summary(text, chunk_words=450, max_len=130, min_len=40):
    """Chunk long text, summarize each chunk, then join."""
    pipe = _get_pipe()
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]

    outputs = []
    for chunk in chunks:
        n = len(chunk.split())
        if n < 30:
            outputs.append(chunk)
            continue
        mx = min(max_len, max(20, n // 2))
        mn = min(min_len, mx - 5)
        result = pipe(chunk, max_length=mx, min_length=mn, do_sample=False, truncation=True)
        outputs.append(result[0]["summary_text"])
    return " ".join(outputs)