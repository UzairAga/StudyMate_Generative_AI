import streamlit as st
import fitz  # PyMuPDF
import faiss
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions

# ---------------- PDF Parsing ----------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---------------- Text Chunking ----------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ---------------- Vector Store ----------------
class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []

    def add_documents(self, texts):
        embeddings = self.model.encode(texts)
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0]]

# ---------------- QA Engine ----------------
class QAModel:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def answer(self, question, context):
        result = self.qa_pipeline(question=question, context=context)
        answer_text = result.get("answer", "")

        # Summarize if answer too long or low confidence
        if len(answer_text.split()) > 50 or result.get("score", 0.0) < 0.4:
            try:
                summary = self.summarizer(answer_text, max_length=80, min_length=20, do_sample=False)
                answer_text = summary[0]["summary_text"]
            except Exception:
                pass

        # Convert to bullet points
        points = [pt.strip() for pt in answer_text.replace("\n", " ").split(".") if pt.strip()]
        result["answer_points"] = points

        return result

    def summarize_context(self, context):
        try:
            summary = self.summarizer(context, max_length=120, min_length=40, do_sample=False)
            points = [pt.strip() for pt in summary[0]["summary_text"].split(".") if pt.strip()]
            return points
        except Exception:
            return [c.strip() for c in context.split(".") if c.strip()]

# ---------------- Watson NLU (Optional) ----------------
def enrich_text(text, api_key, url):
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(version='2021-08-01', authenticator=authenticator)
    nlu.set_service_url(url)
    response = nlu.analyze(text=text, features=Features(keywords=KeywordsOptions(limit=5))).get_result()
    return [kw['text'] for kw in response['keywords']]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="StudyMate", layout="wide")
st.title("📚 StudyMate: AI-Powered PDF Q&A")

uploaded_file = st.file_uploader("Upload your study PDF", type="pdf")
query = st.text_input("Ask a question about the document")

use_watson = st.checkbox("Use IBM Watson NLU for keyword enrichment (optional)")
watson_api_key = st.text_input("IBM Watson API Key", type="password") if use_watson else None
watson_url = st.text_input("IBM Watson URL") if use_watson else None

if uploaded_file and query:
    with st.spinner("Reading and indexing your PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)

        # ✅ Watson runs internally but no output shown
        if use_watson and watson_api_key and watson_url:
            _ = enrich_text(raw_text, watson_api_key, watson_url)

        vs = VectorStore()
        vs.add_documents(chunks)
        relevant_chunks = vs.search(query)
        context = " ".join(relevant_chunks)

        qa = QAModel()
        result = qa.answer(query, context)

        answer_points = result.get("answer_points", [])
        score = result.get("score", 0.0)

        if not answer_points:
            st.warning("⚠️ No confident answers found. Showing summarized context instead:")
            context_points = qa.summarize_context(context)
            for pt in context_points:
                st.write(f"- {pt}")
        else:
            st.success("Answer (Point-wise):")
            for pt in answer_points:
                st.write(f"- {pt}")
            st.caption(f"Confidence: {score:.2f}")
