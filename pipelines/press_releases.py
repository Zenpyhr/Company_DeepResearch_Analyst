from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.logging import get_logger
from app.paths import raw_data_dir
from schemas.models import Company, Source


logger = get_logger(__name__)

# MVP approach: use NVIDIA's newsroom listing page and collect a small set of recent article links.
NVIDIA_NEWSROOM_URL = "https://nvidianews.nvidia.com/news"


def _press_release_dir(ticker: str) -> Path:
    path = raw_data_dir(ticker) / "press_releases"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_press_releases(company: Company, limit: int = 5) -> list[Source]:
    logger.info("Fetching press release listing for %s", company.ticker)
    response = requests.get(NVIDIA_NEWSROOM_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    releases: list[Source] = []
    seen_urls: set[str] = set()
    output_dir = _press_release_dir(company.ticker)

    # Prefer article cards and newsroom links rather than every link on the page.
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if not href:
            continue
        absolute_url = urljoin(NVIDIA_NEWSROOM_URL, href)
        if absolute_url in seen_urls:
            continue
        if not absolute_url.startswith("https://nvidianews.nvidia.com/news/"):
            continue
        if absolute_url.rstrip("/") == NVIDIA_NEWSROOM_URL.rstrip("/"):
            continue

        seen_urls.add(absolute_url)
        article_response = requests.get(absolute_url, timeout=30)
        article_response.raise_for_status()
        article_soup = BeautifulSoup(article_response.text, "html.parser")
        article_title = _extract_title(article_soup) or link.get_text(" ", strip=True) or absolute_url
        article_text = _extract_article_text(article_soup)
        if len(article_text) < 500:
            continue

        file_name = f"press_release_{len(releases) + 1:02d}.txt"
        raw_path = output_dir / file_name
        raw_path.write_text(article_text, encoding="utf-8")

        releases.append(
            Source(
                company_ticker=company.ticker,
                source_type="press_release",
                title=article_title[:200],
                source_url=absolute_url,
                raw_path=str(raw_path),
                metadata_json={"listing_page": NVIDIA_NEWSROOM_URL},
            )
        )
        if len(releases) >= limit:
            break

    return releases


def _extract_title(soup: BeautifulSoup) -> str | None:
    for selector in ("h1", "meta[property='og:title']"):
        element = soup.select_one(selector)
        if element is None:
            continue
        if element.name == "meta":
            content = element.get("content", "").strip()
            if content:
                return content
        else:
            text = element.get_text(" ", strip=True)
            if text:
                return text
    return None


def _extract_article_text(soup: BeautifulSoup) -> str:
    # Remove obvious page chrome before extracting text.
    for selector in ("header", "nav", "footer", "script", "style", ".share", ".social", ".breadcrumbs"):
        for element in soup.select(selector):
            element.decompose()

    candidates = [
        soup.select_one("article"),
        soup.select_one("main"),
        soup.select_one(".article-body"),
        soup.select_one(".entry-content"),
    ]
    for candidate in candidates:
        if candidate is not None:
            text = candidate.get_text("\n", strip=True)
            if len(text) >= 500:
                return text

    return soup.get_text("\n", strip=True)
