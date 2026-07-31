from app.extraction.interests import extract_interests


def test_extract_interests_multiple_categories():
    assert extract_interests("I love hiking and trying local food") == ["outdoors", "food"]


def test_extract_interests_dedupes_same_category():
    assert extract_interests("I love hiking and camping") == ["outdoors"]


def test_extract_interests_returns_empty_list_when_absent():
    assert extract_interests("What is the capital of France?") == []
