
%%writefile search_utils.py

import json
import os
import numpy as np
from typing import List, Dict, Any
from Bio import Entrez
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --- GLOBAL SETUP ---
Entrez.email = "sesimboyle@gmail.com"
print("⏳ Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded.\n")

# --- ROBUST DATA LOADING (Auto-fetches if JSON is missing) ---
def load_or_fetch_pubmed_data(
    filename: str = "pubmed_abstracts.json",
    query: str = "(single-cell OR spatial transcriptomics) AND (human brain) AND (novel cell type)",
    max_records: int = 10
) -> List[Dict[str, Any]]:
    documents = []
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                documents = json.load(f)
            if len(documents) > 0:
                print(f"✅ Loaded {len(documents)} documents from '{filename}'.")
                return documents
        except Exception as e:
            print(f"⚠️ Error reading file: {e}. Will re-fetch.")

    print(f"🔄 Fetching data from PubMed...")
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_records, sort="relevance")
    search_results = Entrez.read(search_handle)
    search_handle.close()
    pmids = search_results.get("IdList", [])

    if not pmids: return []

    fetch_handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    extracted_data = []
    articles_list = records.get('PubmedArticleSet', {}).get('PubmedArticle', [])
    if isinstance(articles_list, dict): articles_list = [articles_list]

    for article in articles_list:
        try:
            pmid_node = article["MedlineCitation"]["PMID"]
            pmid = pmid_node.get("text", "") if isinstance(pmid_node, dict) else str(pmid_node)
            title_raw = article["MedlineCitation"]["Article"]["ArticleTitle"]
            title = title_raw.get("content", "No Title") if isinstance(title_raw, dict) else title_raw
            abstract_node = article["MedlineCitation"]["Article"].get("Abstract", {})
            abstract_text_raw = abstract_node.get("AbstractText", [])

            if isinstance(abstract_text_raw, str): abstract_text = abstract_text_raw
            elif isinstance(abstract_text_raw, list):
                abstract_parts = [part.get('content', '') if isinstance(part, dict) else str(part) for part in abstract_text_raw]
                abstract_text = " ".join(abstract_parts).strip()
            else: abstract_text = ""
            extracted_data.append({"pmid": pmid, "title": title, "abstract": abstract_text})
        except KeyError: continue

    if extracted_data:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2)
        print(f"✅ Successfully fetched and saved {len(extracted_data)} documents!")
    return extracted_data

# --- SEARCH FUNCTIONS ---
def build_bm25_index(documents: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_corpus = [doc["abstract"].lower().split() for doc in documents]
    return BM25Okapi(tokenized_corpus)

def search_bm25(bm25_index: BM25Okapi, documents: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    ranked_results = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [{"score": float(score), "doc": doc} for score, doc in ranked_results[:top_k]]

def generate_embeddings(texts: List[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

def search_vectors(query: str, doc_embeddings: np.ndarray, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query], normalize_embeddings=True)
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [{"score": float(similarities[idx]), "doc": documents[idx]} for idx in top_indices]
