from app.extraction.preferences import QueryPreferences, extract_preferences


def test_extract_preferences_combines_all_three():
    prefs = extract_preferences("Looking for a hotel under $500 in Paris, I love hiking")
    assert prefs == QueryPreferences(city="paris", budget=500.0, interests=["outdoors"])


def test_extract_preferences_all_none_when_nothing_detected():
    # No GPE, no price signal, no interest keyword — unlike "capital of France",
    # which spaCy correctly tags "France" as a GPE (so city would NOT be None there).
    prefs = extract_preferences("What is the square root of 144?")
    assert prefs.city is None
    assert prefs.budget is None
    assert prefs.interests == []
