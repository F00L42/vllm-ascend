"""Build the offline HTML demonstration from the Markdown fact source.

Run from any working directory with Python 3. The HTML template and assets
are resolved relative to this file; the Markdown source is never modified.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = REPORT_DIR / "jev_like_engineering_progress.md"
TEMPLATE_PATH = REPORT_DIR / "assets" / "demo.template.html"


def parse_sections(markdown: str) -> list[dict[str, object]]:
    """Index level-two and level-three headings, including child content."""
    lines = markdown.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_char = ""
    fence_length = 0

    for line_index, line in enumerate(lines):
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                in_fence = False
            continue
        if in_fence:
            continue

        heading = re.match(r"^(#{2,3})[ \t]+(.+?)\s*$", line)
        if heading:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(2))
            headings.append((line_index, len(heading.group(1)), title))

    sections: list[dict[str, object]] = []
    for heading_index, (start, level, title) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _ in headings[heading_index + 1 :]:
            if next_level <= level:
                end = next_start
                break
        sections.append(
            {
                "title": title,
                "level": level,
                "line_start": start + 1,
                "content": "".join(lines[start + 1 : end]).strip(),
            }
        )
    return sections


def extract_links(markdown: str) -> list[dict[str, object]]:
    """Extract distinct HTTP(S) URLs with labels and source line numbers."""
    links: list[dict[str, object]] = []
    seen: set[str] = set()
    markdown_link = re.compile(r"\[([^\]\n]+)\]\(<?(https?://[^\s<>]+?)>?\)")
    bare_url = re.compile(r"https?://[^\s<>\"']+")

    for line_number, line in enumerate(markdown.splitlines(), start=1):
        labelled = list(markdown_link.finditer(line))
        matches: list[tuple[int, str, str]] = [(match.start(), match.group(1), match.group(2)) for match in labelled]
        for match in bare_url.finditer(line):
            if any(item.start() <= match.start() < item.end() for item in labelled):
                continue
            url = match.group(0).rstrip(".,;:!?，。；：！？")
            while url.endswith(")") and url.count(")") > url.count("("):
                url = url[:-1]
            matches.append((match.start(), url, url))

        for _, label, url in sorted(matches, key=lambda item: item[0]):
            if url not in seen:
                seen.add(url)
                links.append({"label": label, "url": url, "line": line_number})
    return links


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build() -> tuple[Path, Path]:
    source_bytes = SOURCE_PATH.read_bytes()
    markdown = source_bytes.decode("utf-8-sig")
    index = {
        "source_file": SOURCE_PATH.name,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sections": parse_sections(markdown),
        "links": extract_links(markdown),
    }
    embedded_source = dict(index, markdown=markdown)
    source_json = json.dumps(embedded_source, ensure_ascii=False, indent=2)
    # JSON lives in a script element. Escape '<' to prevent an embedded source
    # string from closing that element, while preserving the parsed text.
    source_json = source_json.replace("<", "\\u003c")
    source_json = source_json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    replacements = {
        "__SOURCE_JSON__": source_json,
        "__ROUTES_IMAGE__": image_data_url(REPORT_DIR / "ref" / "kev_plumb_routes.png"),
        "__DG_IMAGE__": image_data_url(REPORT_DIR / "ref" / "GD共存分析.png"),
        "__NATIVE_SVG__": (REPORT_DIR / "assets" / "native_serving.svg").read_text(encoding="utf-8-sig"),
    }
    html = TEMPLATE_PATH.read_text(encoding="utf-8-sig")
    missing = [placeholder for placeholder in replacements if placeholder not in html]
    if missing:
        raise ValueError(f"Template is missing placeholders: {', '.join(missing)}")
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    index_path = REPORT_DIR / "source_index.json"
    output_path = REPORT_DIR / "index.html"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path.write_text(html, encoding="utf-8")
    return output_path, index_path


if __name__ == "__main__":
    output_path, index_path = build()
    print(f"HTML: {output_path}")
    print(f"Source index: {index_path}")
