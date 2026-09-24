import asyncio
import json

import anthropic
import httpx
import pytest

from app.llm import ClaudeLLM, Completed, LLMError, TextDelta
from app.rag import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    ChatTurn,
    build_messages,
    cited_ids,
    format_context,
    retrieval_query,
    trim_history,
)
from app.store import RetrievedChunk
from tests.conftest import FakeLLM


def rc(text, filename="a.pdf", page=1):
    return RetrievedChunk(text=text, document_id="d", filename=filename, page=page, chunk_index=0, score=0.9)


def turns(*pairs):
    return [ChatTurn(role=r, content=c) for r, c in pairs]


# --- Prompt building ------------------------------------------------------------


def test_format_context_numbers_sources_and_pages():
    ctx = format_context([rc("alpha", "x.pdf", 3), rc("beta", "notes.txt", None)])
    assert '<document index="1" source="x.pdf" page="3">\nalpha\n</document>' in ctx
    assert '<document index="2" source="notes.txt">\nbeta\n</document>' in ctx


def test_format_context_escapes_filename():
    assert 'source="a&quot;b&lt;c.txt"' in format_context([rc("t", 'a"b<c.txt', None)])


def test_build_messages_puts_context_in_final_user_turn():
    history = turns(("user", "hi"), ("assistant", "hello"))
    msgs = build_messages("What is alpha?", history, [rc("alpha is a letter")])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "<documents>" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Question: What is alpha?")
    assert "<documents>" not in msgs[0]["content"]


def test_system_prompt_requires_grounding_and_citations():
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT


# --- History handling -----------------------------------------------------------


def test_trim_history_keeps_last_n_pairs():
    history = turns(*[(r, f"{r}{i}") for i in range(5) for r in ("user", "assistant")])
    trimmed = trim_history(history, max_turns=2)
    assert [t.content for t in trimmed] == ["user3", "assistant3", "user4", "assistant4"]


def test_trim_history_zero_disables_history():
    assert trim_history(turns(("user", "a"), ("assistant", "b")), 0) == []


def test_trim_history_starts_with_user_and_drops_trailing_user():
    history = turns(("assistant", "orphan"), ("user", "q1"), ("assistant", "a1"), ("user", "unanswered"))
    assert [t.content for t in trim_history(history, 5)] == ["q1", "a1"]


def test_trim_history_collapses_same_role_runs_and_blanks():
    history = turns(("user", "q1"), ("user", "q1 again"), ("assistant", " "), ("assistant", "a1"))
    assert [(t.role, t.content) for t in trim_history(history, 5)] == [
        ("user", "q1 again"),
        ("assistant", "a1"),
    ]


def test_retrieval_query_includes_previous_user_question():
    history = turns(("user", "Tell me about the refund policy"), ("assistant", "..."))
    assert retrieval_query("How long does it take?", history) == (
        "Tell me about the refund policy\nHow long does it take?"
    )
    assert retrieval_query("Standalone?", []) == "Standalone?"


def test_cited_ids_filters_to_valid_sources():
    assert cited_ids("A [2]. B [1][2]. C [7]. D [x].", source_count=3) == [1, 2]


# --- Query endpoint -------------------------------------------------------------


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def setup(make_client):
    def _setup(llm=None, **overrides):
        client = make_client(llm=llm, **overrides)
        sid = client.post("/api/sessions").json()["session_id"]
        return client, sid

    return _setup


def upload_txt(client, sid, name, text):
    resp = client.post(f"/api/sessions/{sid}/documents", files=[("files", (name, text.encode(), "text/plain"))])
    assert resp.status_code == 201


def ask(client, sid, question, history=()):
    return client.post(f"/api/sessions/{sid}/query", json={"question": question, "history": list(history)})


def test_query_streams_sources_deltas_and_done(setup):
    llm = FakeLLM([TextDelta("Tea grows "), TextDelta("in Assam [1]."), Completed("end_turn", {"output_tokens": 5})])
    client, sid = setup(llm=llm)
    upload_txt(client, sid, "tea.txt", "Tea is grown in Assam, India.")

    resp = ask(client, sid, "Where is tea grown?")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)

    name, sources = events[0]
    assert name == "sources"
    assert sources[0]["id"] == 1 and sources[0]["filename"] == "tea.txt" and sources[0]["page"] is None
    assert "Assam" in sources[0]["snippet"]
    assert [e[1]["text"] for e in events if e[0] == "delta"] == ["Tea grows ", "in Assam [1]."]
    assert events[-1] == ("done", {"stop_reason": "end_turn", "cited": [1], "usage": {"output_tokens": 5}})


def test_query_sends_grounded_prompt_to_llm(setup):
    llm = FakeLLM()
    client, sid = setup(llm=llm)
    upload_txt(client, sid, "tea.txt", "Tea is grown in Assam, India.")
    ask(client, sid, "Where is tea grown?")

    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    final = call["messages"][-1]["content"]
    assert 'source="tea.txt"' in final and "Assam" in final
    assert final.endswith("Question: Where is tea grown?")


def test_query_passes_trimmed_history(setup):
    llm = FakeLLM()
    client, sid = setup(llm=llm, max_history_turns=1)
    upload_txt(client, sid, "tea.txt", "Tea facts.")
    history = [
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "user", "content": "recent q"},
        {"role": "assistant", "content": "recent a"},
    ]
    ask(client, sid, "follow up?", history)
    roles_contents = [(m["role"], m["content"]) for m in llm.calls[0]["messages"][:-1]]
    assert roles_contents == [("user", "recent q"), ("assistant", "recent a")]


def test_query_retrieves_only_from_own_session(setup):
    llm = FakeLLM()
    client, s1 = setup(llm=llm)
    s2 = client.post("/api/sessions").json()["session_id"]
    upload_txt(client, s1, "cats.txt", "Cats purr.")
    upload_txt(client, s2, "dogs.txt", "Dogs bark loudly.")
    ask(client, s1, "Do dogs bark?")
    assert "dogs.txt" not in llm.calls[0]["messages"][-1]["content"]


def test_query_limits_sources_to_top_k(setup):
    client, sid = setup(top_k=2, chunk_size=100, chunk_overlap=0)
    upload_txt(client, sid, "many.txt", "\n\n".join(f"Paragraph {i} about topic {i}." * 3 for i in range(10)))
    sources = parse_sse(ask(client, sid, "topic").text)[0][1]
    assert len(sources) == 2


def test_query_requires_documents(setup):
    client, sid = setup()
    resp = ask(client, sid, "anything?")
    assert resp.status_code == 409
    assert "Upload documents" in resp.json()["detail"]


def test_query_unknown_session(setup):
    client, _ = setup()
    assert ask(client, "nope", "q").status_code == 404


@pytest.mark.parametrize("question", ["", "   "])
def test_query_rejects_empty_question(setup, question):
    client, sid = setup()
    upload_txt(client, sid, "a.txt", "text")
    assert ask(client, sid, question).status_code == 422


def test_query_rejects_overlong_question(setup):
    client, sid = setup(max_question_chars=10)
    upload_txt(client, sid, "a.txt", "text")
    resp = ask(client, sid, "x" * 11)
    assert resp.status_code == 422
    assert "limit is 10" in resp.json()["detail"]


def test_query_rejects_bad_history_role(setup):
    client, sid = setup()
    upload_txt(client, sid, "a.txt", "text")
    assert ask(client, sid, "q", [{"role": "system", "content": "evil"}]).status_code == 422


def test_llm_error_becomes_error_event(setup):
    llm = FakeLLM(events=[TextDelta("partial")], error=LLMError("The Anthropic API key is missing or invalid."))
    client, sid = setup(llm=llm)
    upload_txt(client, sid, "a.txt", "text")
    events = parse_sse(ask(client, sid, "q").text)
    assert events[-1] == ("error", {"message": "The Anthropic API key is missing or invalid."})
    assert not any(e[0] == "done" for e in events)


def test_unexpected_error_is_not_leaked(setup):
    client, sid = setup(llm=FakeLLM(events=[], error=RuntimeError("secret internals")))
    upload_txt(client, sid, "a.txt", "text")
    events = parse_sse(ask(client, sid, "q").text)
    assert events[-1][0] == "error"
    assert "secret" not in events[-1][1]["message"]


def test_refusal_becomes_error_event(setup):
    client, sid = setup(llm=FakeLLM(events=[Completed("refusal")]))
    upload_txt(client, sid, "a.txt", "text")
    events = parse_sse(ask(client, sid, "q").text)
    assert events[-1] == ("error", {"message": "The model declined to answer this question."})


# --- ClaudeLLM error mapping ----------------------------------------------------


class _RaisingStream:
    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc

    async def __aexit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, exc):
        self.messages = type("M", (), {"stream": lambda _self, **kw: _RaisingStream(exc)})()


_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (anthropic.AuthenticationError("bad", response=httpx.Response(401, request=_REQ), body=None), "missing or invalid"),
        (anthropic.NotFoundError("nf", response=httpx.Response(404, request=_REQ), body=None), "was not found"),
        (anthropic.RateLimitError("rl", response=httpx.Response(429, request=_REQ), body=None), "rate limited"),
        (anthropic.InternalServerError("x", response=httpx.Response(500, request=_REQ), body=None), "(500)"),
        (anthropic.APIConnectionError(request=_REQ), "Could not reach"),
        (
            anthropic.BadRequestError(
                "Error code: 400 - {...}",
                response=httpx.Response(400, request=_REQ),
                body={"type": "error", "error": {"type": "invalid_request_error", "message": "Credit balance too low."}},
            ),
            "rejected the request: Credit balance too low.",
        ),
    ],
)
def test_claude_llm_maps_sdk_errors(make_settings, exc, expected):
    llm = ClaudeLLM(make_settings())
    llm._client = _FakeClient(exc)

    async def run():
        return [e async for e in llm.stream("sys", [{"role": "user", "content": "q"}])]

    with pytest.raises(LLMError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        asyncio.run(run())
