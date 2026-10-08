from pypdf import PdfReader
from docx import Document


def load_text(uploaded_file) -> str:
    """Read text from a TXT, PDF or DOCX file-like object (Streamlit upload or open file)."""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        data = uploaded_file.read()
        return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Unsupported file type. Use .txt, .pdf or .docx")