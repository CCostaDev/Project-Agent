from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma


STATE_FILE = ".index_state.json"


# ---------------- Embedding provider (Azure OpenAI or OpenAI) -----------------
def get_embeddings():
    """
    Prefer Azure OpenAI if all required AZURE_* vars (including deployment) are set,
    otherwise fall back to standard OpenAI if OPENAI_API_KEY is present.
    """
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_ver = os.getenv("AZURE_OPENAI_API_VERSION")
    az_embed_deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if az_key and az_ep and az_ver and az_embed_deploy:
        from langchain_openai import AzureOpenAIEmbeddings

        print(f"→ Using Azure OpenAI embeddings (deployment='{az_embed_deploy}')")
        return AzureOpenAIEmbeddings(
            api_key=az_key,
            azure_endpoint=az_ep,
            api_version=az_ver,
            deployment=az_embed_deploy,
        )

    std_key = os.getenv("OPENAI_API_KEY")
    if std_key:
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"→ Using OpenAI embeddings (model='{model}')")
        return OpenAIEmbeddings(model=model)

    raise RuntimeError(
        "No embedding provider configured. "
        "Set Azure vars (AZURE_OPENAI_* + AZURE_OPENAI_EMBEDDING_DEPLOYMENT) "
        "or OPENAI_API_KEY."
    )


# ---------------- Helpers -----------------------------------------------------
def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
            h.update(chunk)
    return h.hexdigest()


def load_state() -> Dict[str, str]:
    p = Path(STATE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            # Corrupt state; start fresh
            return {}
    return {}


def save_state(state: Dict[str, str]) -> None:
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


# ---------------- Main --------------------------------------------------------
def main():
    load_dotenv()  # load .env from project root
    docs_dir = Path("sample_docs")
    if not docs_dir.exists():
        raise FileNotFoundError("Create 'sample_docs/' and place at least one text-based PDF inside.")

    chroma_path = os.getenv("CHROMA_DB_PATH", ".chroma")
    embeddings = get_embeddings()

    # Open (or create) the collection once
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    state = load_state()
    changed = 0
    skipped = 0

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for pdf in sorted(docs_dir.glob("*.pdf")):
        key = str(pdf.resolve())
        h = file_hash(pdf)

        if state.get(key) == h:
            print(f"⏩ Skipping (unchanged): {pdf.name}")
            skipped += 1
            continue

        # Load & validate text
        text = load_pdf_text(pdf)
        if not text.strip():
            print(f"⚠️  No extractable text (scanned PDF?): {pdf.name} — skipping.")
            continue

        # Remove any previous chunks for this file (prevents duplicates)
        db.delete(where={"path": key})

        # Create new chunks with metadata for citation
        lc_docs = splitter.create_documents(
            [text],
            metadatas=[{"name": pdf.name, "path": key, "hash": h}],
        )

        # Add to Chroma
        db.add_documents(lc_docs)

        # Update local state
        state[key] = h
        changed += 1
        print(f"🔁 Reindexed: {pdf.name} ({len(lc_docs)} chunks)")

    save_state(state)
    print(f"✅ Done. Changed: {changed}, Skipped: {skipped}. Collection path: {chroma_path}")


if __name__ == "__main__":
    main()
