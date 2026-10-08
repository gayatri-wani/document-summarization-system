import os
import re
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOP = set(ENGLISH_STOP_WORDS)
_STEM = PorterStemmer()
_WORD_RE = re.compile(r"[a-zA-Z]+")

# Keep NLTK data inside the project (avoids the Windows Store path issue)
_LOCAL_NLTK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
os.makedirs(_LOCAL_NLTK, exist_ok=True)
nltk.data.path.insert(0, _LOCAL_NLTK)

_nltk_ready = None  # None = untried, True = works, False = use regex fallback


@lru_cache(maxsize=100_000)
def _stem(word: str) -> str:
    return _STEM.stem(word)


def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _regex_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])', text)
    return [p.strip() for p in parts if p.strip()]


def _split(text: str):
    global _nltk_ready
    from nltk.tokenize import sent_tokenize

    if _nltk_ready is not False:
        try:
            return sent_tokenize(text)
        except Exception:
            if _nltk_ready is None:
                try:
                    nltk.download("punkt_tab", download_dir=_LOCAL_NLTK, quiet=True)
                    nltk.download("punkt", download_dir=_LOCAL_NLTK, quiet=True)
                    result = sent_tokenize(text)
                    _nltk_ready = True
                    return result
                except Exception:
                    pass
            _nltk_ready = False
    return _regex_sentences(text)


def split_sentences(text: str, min_words: int = 5):
    sentences = _split(clean_text(text))
    return [s.strip() for s in sentences if len(s.split()) >= min_words]


def tokenize_pairs(sentence: str):
    """Return [(stem, original_word)] after lowercasing and stopword removal."""
    words = _WORD_RE.findall(sentence.lower())
    return [(_stem(w), w) for w in words if len(w) > 2 and w not in _STOP]


def tokenize(sentence: str):
    return [stem for stem, _ in tokenize_pairs(sentence)]