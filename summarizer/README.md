# Document Summarization System

Information Retrieval mini project. Summarizes TXT/PDF/DOCX documents using
TF-IDF, TextRank, BM25 (query-focused) with MMR, plus an optional transformer.

## Setup
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

## Run
streamlit run app.py