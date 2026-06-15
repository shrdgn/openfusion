"""Landing page tests."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import httpx

from openfusion.server import LANDING_PAGE_DIR


class LandingPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.links: list[dict[str, str]] = []
        self.scripts: list[dict[str, str]] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        if tag == "link":
            self.links.append(attributes)
        if tag == "script":
            self.scripts.append(attributes)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


async def test_root_serves_static_landing_page(client: httpx.AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "OpenAI-compatible compound-model proxy" in response.text
    assert "Start locally" in response.text
    assert 'href="/landing/styles.css"' in response.text

    parser = LandingPageParser()
    parser.feed(response.text)
    assert parser.title == "openfusion - Open compound-model proxy"
    assert parser.scripts == []


async def test_landing_stylesheet_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/landing/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert ":root" in response.text
    assert "@media (max-width: 900px)" in response.text


def test_landing_page_docs_record_security_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs = (repo_root / "docs" / "LANDING_PAGE.md").read_text(encoding="utf-8")

    assert LANDING_PAGE_DIR.joinpath("index.html").is_file()
    assert LANDING_PAGE_DIR.joinpath("styles.css").is_file()
    assert "Provider keys must never appear" in docs
    assert "separate application boundary" in docs
