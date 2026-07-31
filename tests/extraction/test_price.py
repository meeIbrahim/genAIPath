from app.extraction.price import extract_price


def test_extract_price_dollar_amount():
    assert extract_price("Looking for a hotel under $500 in Lahore") == 500.0


def test_extract_price_k_suffix():
    assert extract_price("My budget is 20k") == 20000.0


def test_extract_price_comma_separated():
    assert extract_price("around 2,500 for the trip") == 2500.0


def test_extract_price_returns_none_when_absent():
    assert extract_price("I want a nice hotel in Lahore") is None
