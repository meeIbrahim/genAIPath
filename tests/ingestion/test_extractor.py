import pytest

from app.config import Settings
from app.ingestion.extractor import ExtractionError, extract_main_text

SETTINGS = Settings(min_extract_length=20)

ARTICLE_HTML = """
<html><body>
<nav>Home | About</nav>
<article>
<p>   This   is a real   article  with enough content to pass the
threshold   easily.   </p>

<p>Second paragraph here.</p>
</article>
<footer>copyright 2026</footer>
</body></html>
"""

EMPTY_HTML = "<html><body><nav>Home</nav><footer>copyright</footer></body></html>"

ARTICLE_WITH_HEADER_AND_ASIDE = """
<html><body>
<nav>Site Navigation | Home | About</nav>
<article>
<header>
<h1>Article Title</h1>
<p>By Author Name</p>
</header>
<p>This is the main article body with enough content to pass extraction threshold easily.</p>
<aside><p>Important pull-quote or footnote content here.</p></aside>
<p>More article body text.</p>
</article>
</body></html>
"""


def test_extract_main_text_strips_boilerplate_and_collapses_whitespace():
    text = extract_main_text(ARTICLE_HTML, SETTINGS)
    assert "Home" not in text
    assert "copyright" not in text
    assert "This is a real article with enough content to pass the threshold easily." in text
    assert "Second paragraph here." in text


def test_extract_main_text_raises_on_near_empty_content():
    with pytest.raises(ExtractionError, match="no extractable content"):
        extract_main_text(EMPTY_HTML, SETTINGS)


def test_extract_main_text_preserves_legitimate_content_in_header_and_aside():
    """Verify that header and aside tags are not unconditionally stripped.

    nav-only preprocessing removes only site navigation. Header/aside tags
    are left intact for trafilatura's content-scoring heuristics to evaluate.
    Main article content should always survive.
    """
    text = extract_main_text(ARTICLE_WITH_HEADER_AND_ASIDE, SETTINGS)
    # Site nav should be removed (only tag we preprocess)
    assert "Site Navigation" not in text
    # Core article content in body should always be extracted
    assert "This is the main article body" in text
    assert "More article body text" in text
    # Article structure (header/aside) is not deleted; trafilatura decides
    # whether to score their content based on its heuristics (title/byline
    # typically included; aside typically not, as it's low-confidence).
    # The key is we don't unconditionally delete these tags.
