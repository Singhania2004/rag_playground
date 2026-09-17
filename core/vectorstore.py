from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

_embeddings = None


def get_embeddings():
    """Cache the embedding model so it's only loaded once per process."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def build_vectorstore(docs, collection_name: str = "rag_lab"):
    """Build a fresh in-memory Chroma collection for the current document.

    We intentionally do NOT persist to disk: every time a new document is
    loaded in the app, we want a clean index rather than mixing chunks from
    a previous session.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return vectorstore
