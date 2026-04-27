"""
Knowledge base ingestion script.

Loads all markdown files from data/knowledge_base/, splits them into
chunks, embeds them using sentence-transformers, and persists to ChromaDB.
Run this ONCE before starting the pipeline.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore():
    """
    Load, chunk, embed, and persist the knowledge base into ChromaDB.

    Returns:
        None. Prints progress and final chunk count.
    """
    project_root = Path(__file__).resolve().parent.parent
    kb_path = project_root / "data" / "knowledge_base"
    chroma_path = project_root / "chroma_db"

    print(f"Loading documents from {kb_path}...")

    # Load all .md files
    loader = DirectoryLoader(
        str(kb_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Persist to ChromaDB
    print(f"Building ChromaDB at {chroma_path}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_path),
    )

    print(f"\nIngested {len(chunks)} chunks from {len(documents)} documents into ChromaDB")
    print(f"Vector store persisted to {chroma_path}")


if __name__ == "__main__":
    build_vectorstore()