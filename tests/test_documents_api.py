import pytest

from tests.pdf_utils import make_pdf


@pytest.fixture
def client(make_client):
    return make_client(max_file_size_mb=1, max_files_per_upload=3, max_docs_per_session=4)


@pytest.fixture
def sid(client):
    return client.post("/api/sessions").json()["session_id"]


def txt(name, text):
    return ("files", (name, text.encode(), "text/plain"))


def pdf(name, pages):
    return ("files", (name, make_pdf(pages), "application/pdf"))


def upload(client, sid, *files):
    return client.post(f"/api/sessions/{sid}/documents", files=list(files))


def doc_names(client, sid):
    return [d["filename"] for d in client.get(f"/api/sessions/{sid}/documents").json()["documents"]]


def test_upload_txt_and_pdf(client, sid):
    resp = upload(
        client,
        sid,
        txt("notes.txt", "Tea is grown in Assam."),
        pdf("guide.pdf", ["Page one about coffee.", "Page two about cocoa."]),
    )
    assert resp.status_code == 201
    docs = resp.json()["documents"]
    assert [d["filename"] for d in docs] == ["notes.txt", "guide.pdf"]
    assert docs[0]["file_type"] == "txt" and docs[0]["page_count"] is None
    assert docs[1]["file_type"] == "pdf" and docs[1]["page_count"] == 2
    assert all(d["chunk_count"] >= 1 for d in docs)
    assert doc_names(client, sid) == ["notes.txt", "guide.pdf"]

    hits = client.app.state.store.search(sid, "cocoa", k=1)
    assert hits[0].filename == "guide.pdf" and hits[0].page == 2


def test_documents_accumulate_across_uploads(client, sid):
    upload(client, sid, txt("a.txt", "first"))
    upload(client, sid, txt("b.txt", "second"))
    assert doc_names(client, sid) == ["a.txt", "b.txt"]


def test_unknown_session(client):
    assert upload(client, "nope", txt("a.txt", "x")).status_code == 404
    assert client.get("/api/sessions/nope/documents").status_code == 404


def test_no_files_is_422(client, sid):
    assert client.post(f"/api/sessions/{sid}/documents").status_code == 422


def test_error_response_names_the_file(client, sid):
    resp = upload(client, sid, txt("ok.txt", "fine"), ("files", ("bad.docx", b"x", "application/octet-stream")))
    assert resp.status_code == 415
    assert resp.json()["filename"] == "bad.docx"
    assert "Unsupported file type" in resp.json()["detail"]


def test_batch_is_all_or_nothing(client, sid):
    resp = upload(client, sid, txt("ok.txt", "fine"), ("files", ("fake.pdf", b"not a pdf", "application/pdf")))
    assert resp.status_code == 415
    assert doc_names(client, sid) == []
    assert client.app.state.store.count(sid) == 0


def test_too_many_files_per_upload(client, sid):
    resp = upload(client, sid, *[txt(f"{i}.txt", f"doc {i}") for i in range(4)])
    assert resp.status_code == 400


def test_session_document_limit(client, sid):
    upload(client, sid, *[txt(f"{i}.txt", f"doc {i}") for i in range(3)])
    resp = upload(client, sid, txt("x.txt", "four"), txt("y.txt", "five"))
    assert resp.status_code == 409
    assert "1 more can be added" in resp.json()["detail"]


def test_oversized_file(client, sid):
    resp = upload(client, sid, txt("big.txt", "a" * (1024 * 1024 + 10)))
    assert resp.status_code == 413
    assert resp.json()["filename"] == "big.txt"


def test_oversized_request_rejected_before_parsing(client, sid):
    resp = client.post(
        f"/api/sessions/{sid}/documents",
        content=b"x",
        headers={"content-length": str(50 * 1024 * 1024), "content-type": "multipart/form-data; boundary=b"},
    )
    assert resp.status_code == 413


def test_duplicate_content_rejected(client, sid):
    upload(client, sid, txt("a.txt", "same content"))
    resp = upload(client, sid, txt("renamed.txt", "same content"))
    assert resp.status_code == 409
    assert "'a.txt'" in resp.json()["detail"]


def test_duplicate_within_one_batch_rejected(client, sid):
    resp = upload(client, sid, txt("a.txt", "same"), txt("b.txt", "same"))
    assert resp.status_code == 409
    assert doc_names(client, sid) == []


def test_rollback_when_indexing_fails(client, sid, monkeypatch):
    store = client.app.state.store
    real_add = store.add_chunks
    calls = []

    def flaky_add(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(store, "add_chunks", flaky_add)
    with pytest.raises(RuntimeError):
        upload(client, sid, txt("a.txt", "first doc"), txt("b.txt", "second doc"))
    assert doc_names(client, sid) == []
    assert store.count(sid) == 0


def test_path_components_stripped_from_filename(client, sid):
    resp = upload(client, sid, txt("../../etc/evil.txt", "content"))
    assert resp.json()["documents"][0]["filename"] == "evil.txt"


def test_sessions_do_not_share_documents(client):
    s1 = client.post("/api/sessions").json()["session_id"]
    s2 = client.post("/api/sessions").json()["session_id"]
    upload(client, s1, txt("cats.txt", "cats purr"))
    upload(client, s2, txt("dogs.txt", "dogs bark"))
    assert doc_names(client, s1) == ["cats.txt"]
    assert doc_names(client, s2) == ["dogs.txt"]
    assert {h.filename for h in client.app.state.store.search(s1, "dogs bark", k=5)} == {"cats.txt"}


def test_deleting_session_removes_documents_and_vectors(client, sid):
    upload(client, sid, txt("a.txt", "alpha"))
    client.delete(f"/api/sessions/{sid}")
    assert client.app.state.sessions.list_documents(sid) == []
    assert client.app.state.store.count(sid) == 0


def test_purge_removes_documents_and_vectors(client, sid, monkeypatch):
    upload(client, sid, txt("a.txt", "alpha"))
    sessions = client.app.state.sessions
    monkeypatch.setattr(sessions, "_ttl", -1)  # everything is now expired
    assert client.app.state.purge_expired_sessions() == [sid]
    assert sessions.list_documents(sid) == []
    assert client.app.state.store.count(sid) == 0
