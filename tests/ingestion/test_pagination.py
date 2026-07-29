from app.ingestion.pagination import detect_pagination, detect_next_url, render_template

RENDER_NEXT_LINK = """
<html><head><link rel="next" href="/blog/post-1?page=2"></head><body>x</body></html>
"""

ANCHOR_NEXT_TEXT = """
<html><body><a href="/blog/post-1/2">Next</a></body></html>
"""

ANCHOR_NEXT_DIFFERENT_DOMAIN = """
<html><body><a href="https://other.com/2">Next</a></body></html>
"""

NUMBERED_CLUSTER = """
<html><body>
<a href="/blog?page=2">2</a>
<a href="/blog?page=3">3</a>
<a href="/blog?page=4">4</a>
</body></html>
"""

NO_PAGINATION = "<html><body><p>just an article</p></body></html>"


def test_link_rel_next_wins_top_priority():
    plan = detect_pagination(RENDER_NEXT_LINK, "https://example.com/blog/post-1")
    assert plan.mode == "chain"
    assert plan.next_url == "https://example.com/blog/post-1?page=2"


def test_anchor_next_text_same_path_prefix():
    url = detect_next_url(ANCHOR_NEXT_TEXT, "https://example.com/blog/post-1")
    assert url == "https://example.com/blog/post-1/2"


def test_anchor_next_different_domain_rejected():
    url = detect_next_url(ANCHOR_NEXT_DIFFERENT_DOMAIN, "https://example.com/blog/post-1")
    assert url is None


def test_numbered_cluster_detected_as_template():
    plan = detect_pagination(NUMBERED_CLUSTER, "https://example.com/blog")
    assert plan.mode == "template"
    assert plan.start_page_number == 2
    assert render_template(plan.template, 2) == "https://example.com/blog?page=2"
    assert render_template(plan.template, 3) == "https://example.com/blog?page=3"


def test_no_pagination_signal_is_single():
    plan = detect_pagination(NO_PAGINATION, "https://example.com/blog")
    assert plan.mode == "single"
    assert plan.next_url is None
    assert plan.template is None


def test_anchor_next_different_resource_rejected():
    """Reject same-domain links that share only a string prefix, not a path segment boundary."""
    html = """
    <html><body><a href="/blog/post-123/comments">Next</a></body></html>
    """
    url = detect_next_url(html, "https://example.com/blog/post-1")
    assert url is None  # /blog/post-123/comments does not start with /blog/post-1/


def test_anchor_next_similar_path_different_segment_rejected():
    """Reject links that share a string prefix but diverge at segment boundary."""
    html = """
    <html><body><a href="/blogger/page2">Next</a></body></html>
    """
    url = detect_next_url(html, "https://example.com/blog")
    assert url is None  # /blogger is not a child of /blog


def test_detect_pagination_with_current_url_page_number():
    """Current URL already contains a page number; start_page_number should be N+1."""
    html = """
    <html><body>
    <a href="/blog?page=4">4</a>
    <a href="/blog?page=5">5</a>
    <a href="/blog?page=6">6</a>
    </body></html>
    """
    plan = detect_pagination(html, "https://example.com/blog?page=3")
    assert plan.mode == "template"
    assert plan.start_page_number == 4  # Next page after current page 3
    assert render_template(plan.template, 4) == "https://example.com/blog?page=4"
