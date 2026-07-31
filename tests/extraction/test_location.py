from app.extraction.location import extract_city


def test_extract_city_finds_gpe_entity():
    assert extract_city("I want to visit Paris next month") == "paris"


def test_extract_city_finds_first_of_multiple():
    assert extract_city("Flying from London to Tokyo") == "london"


def test_extract_city_returns_none_when_absent():
    assert extract_city("I like hiking and cheap food") is None


def test_extract_city_finds_lahore():
    # Exercised again in Task 5's router integration test — confirmed here first,
    # at the unit level, so a spaCy GPE-tagging surprise fails fast in this task.
    assert extract_city("A hotel in Lahore") == "lahore"
