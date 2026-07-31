from app.extraction.interests import extract_interests


def test_extract_interests_multiple_categories():
    assert extract_interests("I love hiking and trying local food") == ["outdoors", "food"]


def test_extract_interests_dedupes_same_category():
    assert extract_interests("I love hiking and camping") == ["outdoors"]


def test_extract_interests_returns_empty_list_when_absent():
    assert extract_interests("What is the capital of France?") == []


def test_extract_interests_no_false_positive_on_substring_of_art():
    # Regression: word boundaries prevent "art" from matching inside "apartment", "started", "smart"
    assert extract_interests("Looking for an apartment in Lahore") == []
    assert extract_interests("I started my trip yesterday") == []
    assert extract_interests("smart budget travel tips") == []
