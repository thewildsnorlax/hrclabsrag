import pytest

from app.ingestion import Chunk
from app.store import SentenceTransformerEmbedder, VectorStore
from tests.conftest import FakeEmbedder


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", FakeEmbedder())


def chunks(*texts, page=None):
    return [Chunk(text=t, page=page, index=i) for i, t in enumerate(texts)]


def test_search_empty_session_returns_nothing(store):
    assert store.search("s1", "anything", k=5) == []


def test_search_ranks_relevant_chunk_first(store):
    store.add_chunks(
        "s1",
        "doc1",
        "zoo.txt",
        chunks(
            "Elephants are the largest land animals.",
            "Penguins cannot fly but swim well.",
            "Giraffes have very long necks.",
        ),
    )
    results = store.search("s1", "which animals swim well", k=3)
    assert results[0].text.startswith("Penguins")
    assert results[0].filename == "zoo.txt"
    assert results[0].document_id == "doc1"
    assert results[0].score > results[-1].score


def test_search_returns_at_most_k(store):
    store.add_chunks("s1", "doc1", "a.txt", chunks(*[f"chunk number {i}" for i in range(10)]))
    assert len(store.search("s1", "chunk", k=3)) == 3


def test_page_metadata_round_trips(store):
    store.add_chunks("s1", "pdf", "a.pdf", chunks("text on page seven", page=7))
    store.add_chunks("s1", "txt", "b.txt", chunks("plain text without pages"))
    by_file = {r.filename: r for r in store.search("s1", "text", k=5)}
    assert by_file["a.pdf"].page == 7
    assert by_file["b.txt"].page is None


def test_sessions_are_isolated(store):
    store.add_chunks("s1", "d1", "cats.txt", chunks("Cats purr when content."))
    store.add_chunks("s2", "d2", "rockets.txt", chunks("Rockets burn liquid fuel."))
    assert {r.filename for r in store.search("s1", "rockets fuel", k=5)} == {"cats.txt"}
    assert {r.filename for r in store.search("s2", "cats purr", k=5)} == {"rockets.txt"}


def test_delete_document(store):
    store.add_chunks("s1", "d1", "a.txt", chunks("alpha one", "alpha two"))
    store.add_chunks("s1", "d2", "b.txt", chunks("beta"))
    store.delete_document("s1", "d1")
    assert store.count("s1") == 1


def test_delete_session_is_idempotent(store):
    store.add_chunks("s1", "d1", "a.txt", chunks("alpha"))
    store.delete_session("s1")
    store.delete_session("s1")
    assert store.search("s1", "alpha", k=5) == []


def test_vectors_persist_across_reopen(tmp_path):
    VectorStore(tmp_path / "chroma", FakeEmbedder()).add_chunks("s1", "d1", "a.txt", chunks("kept"))
    assert VectorStore(tmp_path / "chroma", FakeEmbedder()).count("s1") == 1


@pytest.mark.slow
def test_real_embedder_retrieves_semantically(tmp_path):
    """No word overlap between query and answer: only real embeddings can match them."""
    store = VectorStore(tmp_path / "chroma", SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"))
    store.add_chunks(
        "s1",
        "d1",
        "mixed.txt",
        chunks(
            "The quarterly revenue grew by twelve percent year over year.",
            "Employees may work remotely up to three days per week.",
            "The server cluster runs on Kubernetes in two regions.",
        ),
    )
    top = store.search("s1", "What is the work from home policy?", k=3)[0]
    assert top.text.startswith("Employees")
