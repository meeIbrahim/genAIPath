from app.indexing.chunker import chunk_text, sentence_spans

WORD_COUNTER = lambda text: len(text.split())  # noqa: E731 — predictable counts for these tests


def test_sentence_spans_splits_on_terminal_punctuation():
    text = "First sentence. Second sentence! Third one?"
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
    ]


def test_chunk_text_packs_sentences_within_token_budget():
    text = "one two three. four five six. seven eight nine. ten eleven twelve."
    chunks = chunk_text(text, chunk_size=6, overlap=0, token_counter=WORD_COUNTER)
    assert [c.text for c in chunks] == [
        "one two three. four five six.",
        "seven eight nine. ten eleven twelve.",
    ]
    assert chunks[0].char_start == 0
    assert chunks[1].char_start == text.index("seven")


def test_chunk_text_applies_overlap_between_windows():
    text = "one two. three four. five six. seven eight."
    chunks = chunk_text(text, chunk_size=4, overlap=2, token_counter=WORD_COUNTER)
    assert len(chunks) >= 2
    assert chunks[0].overlap_with_prev == 0
    assert chunks[1].overlap_with_prev > 0
    # the overlapping words from the end of chunk 0 reappear at the start of chunk 1
    assert chunks[0].text.split(".")[-2].strip() in chunks[1].text


def test_chunk_text_oversized_single_sentence_stands_alone():
    text = "one two three four five six seven eight nine ten."
    chunks = chunk_text(text, chunk_size=3, overlap=0, token_counter=WORD_COUNTER)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("", chunk_size=400, overlap=75) == []


def test_count_tokens_default_uses_tiktoken():
    from app.indexing.chunker import count_tokens
    assert count_tokens("hello world") > 0
