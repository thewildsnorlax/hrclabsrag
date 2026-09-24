import io

import pytest
from pypdf import PdfWriter

from app.ingestion import process_file
from app.ingestion.chunker import split_text
from app.ingestion.errors import (
    FileTooLarge,
    SessionDocumentLimit,
    TooManyFiles,
    TooManyPages,
    UnreadableDocument,
    UnsupportedFileType,
)
from app.ingestion.loaders import normalize_text
from app.ingestion.validation import clean_filename, validate_batch
from tests.pdf_utils import make_pdf


@pytest.fixture
def settings(make_settings):
    return make_settings(
        max_file_size_mb=1,
        max_pdf_pages=3,
        max_files_per_upload=2,
        max_docs_per_session=3,
        chunk_size=200,
        chunk_overlap=40,
    )


def blank_pdf(pages=1, password=None) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    if password:
        writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# --- Batch limits -----------------------------------------------------------


def test_batch_within_limits(settings):
    validate_batch(file_count=2, existing_doc_count=1, settings=settings)


def test_batch_rejects_empty_upload(settings):
    with pytest.raises(TooManyFiles):
        validate_batch(0, 0, settings)


def test_batch_rejects_too_many_files(settings):
    with pytest.raises(TooManyFiles, match="limit is 2"):
        validate_batch(3, 0, settings)


def test_batch_rejects_exceeding_session_doc_limit(settings):
    with pytest.raises(SessionDocumentLimit, match="1 more can be added") as exc:
        validate_batch(2, 2, settings)
    assert exc.value.status_code == 409


# --- Per-file validation ----------------------------------------------------


@pytest.mark.parametrize("name", ["notes.docx", "page.html", "README.md", "noext"])
def test_rejects_unsupported_types(settings, name):
    with pytest.raises(UnsupportedFileType) as exc:
        process_file(name, b"hello", settings)
    assert exc.value.status_code == 415


def test_rejects_empty_file(settings):
    with pytest.raises(UnsupportedFileType, match="empty"):
        process_file("a.txt", b"", settings)


def test_rejects_oversized_file(settings):
    with pytest.raises(FileTooLarge, match="limit is 1 MB") as exc:
        process_file("big.txt", b"a" * (1024 * 1024 + 1), settings)
    assert exc.value.status_code == 413


def test_rejects_fake_pdf(settings):
    with pytest.raises(UnsupportedFileType, match="not a PDF"):
        process_file("fake.pdf", b"just some text", settings)


def test_rejects_binary_txt(settings):
    with pytest.raises(UnsupportedFileType, match="binary"):
        process_file("bin.txt", b"\x89PNG\x00\x00\x01", settings)


def test_extension_check_is_case_insensitive(settings):
    assert process_file("NOTES.TXT", b"hello world", settings).document.file_type == "txt"


@pytest.mark.parametrize(
    "raw, expected",
    [("../../etc/passwd.txt", "passwd.txt"), ("C:\\docs\\a.pdf", "a.pdf"), ("", "untitled")],
)
def test_clean_filename(raw, expected):
    assert clean_filename(raw) == expected


# --- TXT loading --------------------------------------------------------------


def test_txt_utf8(settings):
    result = process_file("u.txt", "Café naïve résumé".encode("utf-8"), settings)
    assert result.document.pages[0].text == "Café naïve résumé"
    assert result.document.pages[0].number is None
    assert result.document.page_count is None


def test_txt_utf8_bom_is_stripped(settings):
    result = process_file("bom.txt", b"\xef\xbb\xbfhello", settings)
    assert result.document.pages[0].text == "hello"


def test_txt_latin1_fallback(settings):
    result = process_file("l.txt", "Café".encode("latin-1"), settings)
    assert result.document.pages[0].text == "Café"


def test_whitespace_only_txt_is_unreadable(settings):
    with pytest.raises(UnreadableDocument):
        process_file("blank.txt", b"   \n\n\t  ", settings)


# --- PDF loading --------------------------------------------------------------


def test_pdf_pages_keep_numbers(settings):
    data = make_pdf(["The first page talks about apples.", "The second page talks about pears."])
    result = process_file("fruit.pdf", data, settings)
    doc = result.document
    assert doc.file_type == "pdf"
    assert doc.page_count == 2
    assert [p.number for p in doc.pages] == [1, 2]
    assert "apples" in doc.pages[0].text and "pears" in doc.pages[1].text
    assert {c.page for c in result.chunks} == {1, 2}


def test_pdf_page_limit(settings):
    with pytest.raises(TooManyPages, match="4 pages; the limit is 3") as exc:
        process_file("long.pdf", make_pdf(["p"] * 4), settings)
    assert exc.value.status_code == 413


def test_pdf_without_text_is_unreadable(settings):
    with pytest.raises(UnreadableDocument, match="scanned PDF") as exc:
        process_file("scan.pdf", blank_pdf(), settings)
    assert exc.value.status_code == 422


def test_password_protected_pdf_is_unreadable(settings):
    with pytest.raises(UnreadableDocument, match="password"):
        process_file("secret.pdf", blank_pdf(password="pw"), settings)


def test_corrupt_pdf_is_unreadable(settings):
    with pytest.raises(UnreadableDocument):
        process_file("broken.pdf", b"%PDF-1.4\n garbage garbage", settings)


# --- Chunking -----------------------------------------------------------------


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def test_short_text_is_single_chunk():
    assert split_text("Just one sentence.", 200, 40) == ["Just one sentence."]


def test_empty_text_has_no_chunks():
    assert split_text("  \n ", 200, 40) == []


def test_chunks_respect_size_limit():
    text = "\n\n".join(_words(60) for _ in range(5))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_cover_all_content():
    text = _words(300)
    chunks = split_text(text, 200, 40)
    joined = " ".join(chunks)
    assert all(f"word{i}" in joined.split() for i in range(300))


def test_consecutive_chunks_overlap():
    chunks = split_text(_words(300), 200, 40)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in nxt.split()  # tail of one chunk carried into the next


def test_zero_overlap_has_no_duplication():
    chunks = split_text(_words(300), 200, 0)
    words = " ".join(chunks).split()
    assert len(words) == len(set(words)) == 300


def test_prefers_paragraph_boundaries():
    para_a, para_b = "Alpha " * 20, "Beta " * 20  # ~120 and ~100 chars
    chunks = split_text(f"{para_a.strip()}\n\n{para_b.strip()}", 150, 0)
    assert chunks == [para_a.strip(), para_b.strip()]


def test_unbreakable_text_is_hard_split():
    chunks = split_text("x" * 450, 200, 0)
    assert [len(c) for c in chunks] == [200, 200, 50]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_chunk_indexes_are_sequential_across_pages(settings):
    data = make_pdf([_words(80), _words(80)])
    chunks = process_file("x.pdf", data, settings).chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))


# --- Normalisation ------------------------------------------------------------


def test_normalize_text():
    raw = "Line one   \r\nLine\x00 two\n\n\n\n\nPara  two\t\tend  "
    assert normalize_text(raw) == "Line one\nLine two\n\nPara two end"
