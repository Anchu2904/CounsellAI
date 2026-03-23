"""
ingest.py – CounsellAI
One-time (or refresh) script that:
  1. Scans ./data/ recursively for every PDF and CSV
  2. Loads each file with the correct LangChain loader
  3. Splits text into overlapping chunks
  4. Enriches every chunk with metadata (category, country, source, page)
  5. Embeds with BGE-large-en-v1.5 (local, free)
  6. Persists to Chroma at ./chroma_db

Run:  python ingest.py
      python ingest.py --reset   ← wipes chroma_db and re-ingests everything
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("./data")
CHROMA_DIR      = "./chroma_db"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
COLLECTION_NAME = "counsellai"

# ── Keyword maps for metadata auto-detection ───────────────────────────────────
CATEGORY_KEYWORDS = {
    "academic":   ["syllabus", "curriculum", "study", "subject", "board", "class",
                   "marks", "grade", "exam", "jee", "neet", "cbse", "icse", "10th",
                   "12th", "learning", "textbook", "course", "stream"],
    "career":     ["career", "placement", "job", "salary", "aptitude", "skill",
                   "vocation", "profession", "internship", "resume", "interview",
                   "engineering", "medicine", "law", "arts", "commerce"],
    "personal":   ["mental health", "stress", "anxiety", "wellbeing", "wellness",
                   "counseling", "counselling", "emotion", "motivation", "mindset",
                   "pressure", "family", "peer", "bullying", "depression"],
    "admissions": ["admission", "application", "university", "college", "entrance",
                   "shortlist", "scholarship", "deadline", "ielts", "toefl",
                   "sat", "gre", "gmat", "visa", "rank", "cutoff", "nri"],
}

COUNTRY_KEYWORDS = {
    "USA":       ["usa", "united states", "american university", "f-1", "sat",
                  "common app", "ivy league", "act"],
    "UK":        ["uk", "united kingdom", "ucas", "russell group", "british",
                  "tier 4", "student route visa"],
    "Canada":    ["canada", "canadian", "ircc", "sds", "pgwp"],
    "Australia": ["australia", "australian", "cricos", "ielts australia"],
    "Germany":   ["germany", "german", "daad", "studienkolleg", "blocked account"],
    "India":     ["india", "indian", "jee", "neet", "cbse", "icse", "du", "iit",
                  "nit", "bits", "iim", "cat", "upsc", "state board"],
    "Global":    [],   # fallback
}


def detect_category(filename: str, text_sample: str) -> str:
    """Heuristic category detection from filename + text sample."""
    combined = (filename + " " + text_sample).lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "academic"


def detect_country(filename: str, text_sample: str) -> str:
    """Heuristic country detection from filename + text sample."""
    combined = (filename + " " + text_sample).lower()
    scores = {country: 0 for country in COUNTRY_KEYWORDS}
    for country, keywords in COUNTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[country] += 1
    # Remove fallback from scoring
    scores.pop("Global", None)
    if not scores or max(scores.values()) == 0:
        return "Global"
    return max(scores, key=scores.get)


def load_pdf(filepath: Path) -> List[Document]:
    """Load a PDF and return list of Documents (one per page)."""
    try:
        loader = PyPDFLoader(str(filepath))
        docs = loader.load()
        print(f"  ✔ PDF loaded  → {len(docs)} pages  [{filepath.name}]")
        return docs
    except Exception as e:
        print(f"  ✘ Failed to load PDF {filepath.name}: {e}")
        return []


def load_csv(filepath: Path) -> List[Document]:
    """Load a CSV and return list of Documents (one per row)."""
    try:
        loader = CSVLoader(
            file_path=str(filepath),
            csv_args={"delimiter": ",", "quotechar": '"'},
        )
        docs = loader.load()
        print(f"  ✔ CSV loaded  → {len(docs)} rows  [{filepath.name}]")
        return docs
    except Exception as e:
        # Fallback: try tab-separated
        try:
            loader = CSVLoader(
                file_path=str(filepath),
                csv_args={"delimiter": "\t"},
            )
            docs = loader.load()
            print(f"  ✔ TSV loaded  → {len(docs)} rows  [{filepath.name}]")
            return docs
        except Exception as e2:
            print(f"  ✘ Failed to load CSV {filepath.name}: {e2}")
            return []


def enrich_metadata(docs: List[Document], filepath: Path) -> List[Document]:
    """
    Add or override metadata on every Document chunk:
      - source   : relative path from DATA_DIR
      - category : academic | career | personal | admissions
      - country  : India | Global | USA | UK | Canada | Australia | Germany
      - page     : page/row number (if already set by loader, keep it)
    """
    filename = filepath.name
    # Use first 400 chars of first doc as sample for heuristics
    sample_text = docs[0].page_content[:400] if docs else ""

    category = detect_category(filename, sample_text)
    country  = detect_country(filename, sample_text)

    for doc in docs:
        doc.metadata.setdefault("page", 0)          # keep loader's page if present
        doc.metadata["source"]   = str(filepath.relative_to(DATA_DIR.parent))
        doc.metadata["filename"] = filename
        doc.metadata["category"] = category
        doc.metadata["country"]  = country

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def scan_data_dir() -> List[Tuple[Path, str]]:
    """Return list of (path, file_type) tuples for all supported files."""
    supported = []
    if not DATA_DIR.exists():
        print(f"⚠  Data directory '{DATA_DIR}' not found. Creating it.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("   Place your PDF and CSV files inside ./data/ and re-run ingest.py")
        return []

    for fp in sorted(DATA_DIR.rglob("*")):
        if fp.is_file():
            ext = fp.suffix.lower()
            if ext == ".pdf":
                supported.append((fp, "pdf"))
            elif ext in (".csv", ".tsv"):
                supported.append((fp, "csv"))
            else:
                print(f"  ⚠  Skipping unsupported file: {fp.name}")
    return supported


def build_vectorstore(all_chunks: List[Document]) -> Chroma:
    print(f"\n🔢  Loading embedding model: {EMBED_MODEL} …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # cosine-ready
    )

    print(f"💾  Persisting {len(all_chunks)} chunks → {CHROMA_DIR} …")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def print_summary(all_chunks: List[Document]) -> None:
    from collections import Counter
    cats     = Counter(c.metadata.get("category", "?") for c in all_chunks)
    countries = Counter(c.metadata.get("country", "?")  for c in all_chunks)
    sources  = Counter(c.metadata.get("filename", "?")  for c in all_chunks)

    print("\n" + "═" * 60)
    print("  INGESTION SUMMARY")
    print("═" * 60)
    print(f"  Total chunks ingested : {len(all_chunks)}")
    print(f"\n  By category:")
    for cat, n in cats.most_common():
        print(f"    {cat:<15} {n:>5} chunks")
    print(f"\n  By country:")
    for cty, n in countries.most_common():
        print(f"    {cty:<15} {n:>5} chunks")
    print(f"\n  By source file:")
    for src, n in sources.most_common():
        print(f"    {src:<40} {n:>5} chunks")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(description="CounsellAI – Data Ingestion")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing chroma_db and re-ingest everything from scratch."
    )
    args = parser.parse_args()

    # Optional reset
    if args.reset and Path(CHROMA_DIR).exists():
        print(f"🗑  Resetting Chroma store at {CHROMA_DIR} …")
        shutil.rmtree(CHROMA_DIR)

    print("=" * 60)
    print("  CounsellAI – Data Ingestion Pipeline")
    print("=" * 60)

    files = scan_data_dir()
    if not files:
        print("\n⚠  No PDF or CSV files found in ./data/. Exiting.")
        sys.exit(0)

    print(f"\n📂  Found {len(files)} file(s) in ./data/\n")

    all_chunks: List[Document] = []

    for filepath, ftype in files:
        print(f"📄  Processing: {filepath.name}")

        # Load
        if ftype == "pdf":
            raw_docs = load_pdf(filepath)
        else:
            raw_docs = load_csv(filepath)

        if not raw_docs:
            continue

        # Enrich metadata
        raw_docs = enrich_metadata(raw_docs, filepath)

        # Split
        chunks = split_documents(raw_docs)
        print(f"  ✂  Split into {len(chunks)} chunks")

        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n⚠  No chunks produced. Check your files and try again.")
        sys.exit(1)

    # Build + persist vectorstore
    vectorstore = build_vectorstore(all_chunks)

    print_summary(all_chunks)
    print(f"\n✅  Ingestion complete! Chroma DB saved to: {CHROMA_DIR}")
    print("    You can now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
