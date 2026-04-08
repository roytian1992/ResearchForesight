from __future__ import annotations

import io
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from bs4 import BeautifulSoup


SECTION_CMD_RE = re.compile(
    r"\\(?P<cmd>part|chapter|section|subsection|subsubsection|paragraph)\*?"
    r"(?:\[[^\]]*\])?\{(?P<title>[^{}]+)\}"
)
INPUT_RE = re.compile(r"\\(?:input|include)\{([^{}]+)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)
COMMAND_RE = re.compile(r"\\[a-zA-Z@]+(?:\*?)")
BRACE_ARG_RE = re.compile(r"\{[^{}]*\}")
WHITESPACE_RE = re.compile(r"\s+")

SECTION_LEVEL = {
    "part": 0,
    "chapter": 0,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
}

DEFAULT_HEADERS = {
    "User-Agent": "ResearchWorld/0.1 (+SciVisionBench full-text fetcher)",
}


def safe_text(text: str, limit: int = 1200) -> str:
    normalized = WHITESPACE_RE.sub(" ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def fetch_url_bytes(url: str, *, timeout: int = 60) -> Tuple[bytes, str]:
    response = requests.get(url, timeout=timeout, headers=DEFAULT_HEADERS, allow_redirects=True)
    response.raise_for_status()
    return response.content, str(response.headers.get("content-type") or "")


def arxiv_source_url(source_paper_id: str) -> str:
    return f"https://arxiv.org/e-print/{source_paper_id}"


def arxiv_html_url(source_paper_id: str) -> str:
    return f"https://arxiv.org/html/{source_paper_id}"


def arxiv_pdf_url(source_paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{source_paper_id}.pdf"


def extract_archive_files(blob: bytes) -> Dict[str, str]:
    files: Dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            suffix = Path(name).suffix.lower()
            if suffix not in {".tex", ".bbl", ".txt"}:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
            files[name] = text
    return files


def choose_main_tex_file(files: Dict[str, str]) -> Tuple[str, str]:
    scored: List[Tuple[int, int, str, str]] = []
    for name, text in files.items():
        if not name.lower().endswith(".tex"):
            continue
        score = 0
        lowered_name = name.lower()
        lowered_text = text.lower()
        if "\\begin{document}" in lowered_text:
            score += 100
        if "\\title" in lowered_text:
            score += 15
        if "\\abstract" in lowered_text or "\\begin{abstract}" in lowered_text:
            score += 10
        if "main" in lowered_name:
            score += 8
        scored.append((score, len(text), name, text))
    if not scored:
        raise ValueError("No TeX file found in source archive")
    scored.sort(reverse=True)
    _, _, name, text = scored[0]
    return name, text


def inline_tex_inputs(text: str, files: Dict[str, str], *, max_depth: int = 8) -> str:
    def _resolve(body: str, depth: int, seen: set[str]) -> str:
        if depth > max_depth:
            return body

        def repl(match: re.Match) -> str:
            target = match.group(1).strip()
            if not target:
                return ""
            candidate_names = [target]
            if not target.endswith(".tex"):
                candidate_names.append(f"{target}.tex")
            for name in candidate_names:
                basename = Path(name).name
                if basename in seen or basename not in files:
                    continue
                seen.add(basename)
                return _resolve(files[basename], depth + 1, seen)
            return ""

        return INPUT_RE.sub(repl, body)

    return _resolve(text, 0, set())


def tex_to_plain_text(text: str) -> str:
    body = COMMENT_RE.sub("", text)
    body = re.sub(r"\\begin\{abstract\}", "\n\\section{Abstract}\n", body)
    body = re.sub(r"\\end\{abstract\}", "\n", body)
    body = re.sub(r"\\begin\{(figure|table)\*?\}", "\n", body)
    body = re.sub(r"\\end\{(figure|table)\*?\}", "\n", body)
    body = re.sub(r"\\caption\{([^{}]*)\}", r"\nCaption: \1\n", body)
    body = re.sub(r"\\label\{[^{}]*\}", "", body)
    body = COMMAND_RE.sub(" ", body)
    prev = None
    while prev != body:
        prev = body
        body = BRACE_ARG_RE.sub(lambda m: " " + m.group(0)[1:-1] + " ", body)
    body = body.replace("{", " ").replace("}", " ")
    body = body.replace("~", " ")
    return WHITESPACE_RE.sub(" ", body).strip()


def tex_to_sections(main_name: str, text: str) -> List[Dict]:
    markers: List[Tuple[int, int, str, int]] = []
    for match in SECTION_CMD_RE.finditer(text):
        cmd = match.group("cmd")
        title = safe_text(match.group("title"), limit=200)
        level = SECTION_LEVEL.get(cmd, 2)
        markers.append((match.start(), match.end(), title, level))

    if not markers:
        plain = safe_text(tex_to_plain_text(text), limit=4000)
        return [
            {
                "section_id": f"{Path(main_name).stem}::body",
                "title": "Body",
                "level": 1,
                "text": plain,
            }
        ]

    sections: List[Dict] = []
    preamble = text[: markers[0][0]]
    preamble_text = safe_text(tex_to_plain_text(preamble), limit=2000)
    if preamble_text:
        sections.append(
            {
                "section_id": f"{Path(main_name).stem}::abstract",
                "title": "Abstract",
                "level": 1,
                "text": preamble_text,
            }
        )

    for idx, (_, end, title, level) in enumerate(markers):
        next_start = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        chunk = text[end:next_start]
        plain = safe_text(tex_to_plain_text(chunk), limit=4000)
        if not plain:
            continue
        sections.append(
            {
                "section_id": f"{Path(main_name).stem}::sec_{idx + 1}",
                "title": title,
                "level": level,
                "text": plain,
            }
        )
    return sections


def html_to_sections(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    content = soup.find("main") or soup.find("article") or soup.body or soup
    sections: List[Dict] = []
    current_title = "Body"
    current_level = 1
    current_texts: List[str] = []
    section_idx = 0
    title_tag = soup.find("title")
    page_title = safe_text(title_tag.get_text(" ", strip=True), limit=200) if title_tag else ""

    def flush() -> None:
        nonlocal section_idx, current_texts
        text = safe_text(" ".join(current_texts), limit=4000)
        if not text:
            current_texts = []
            return
        section_idx += 1
        sections.append(
            {
                "section_id": f"html::sec_{section_idx}",
                "title": current_title,
                "level": current_level,
                "text": text,
            }
        )
        current_texts = []

    for node in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "figcaption", "caption"], recursive=True):
        if node.name.startswith("h"):
            heading_text = safe_text(node.get_text(" ", strip=True), limit=200) or "Section"
            if (
                not sections
                and not current_texts
                and page_title
                and heading_text.lower() == page_title.lower()
            ):
                continue
            flush()
            current_title = heading_text
            try:
                current_level = int(node.name[1])
            except Exception:
                current_level = 2
            continue
        text = safe_text(node.get_text(" ", strip=True), limit=800)
        if text:
            current_texts.append(text)
    flush()
    return sections


def fallback_abstract_sections(abstract: str) -> List[Dict]:
    text = safe_text(abstract, limit=2000)
    if not text:
        return []
    return [
        {
            "section_id": "abstract::only",
            "title": "Abstract",
            "level": 1,
            "text": text,
        }
    ]


PDF_HEADING_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[\s.\-:]+(.+?)\s*$")
PDF_HEADING_APPENDIX_RE = re.compile(r"^\s*(appendix|references|acknowledg(?:e)?ments?|conclusion|conclusions)\s*$", re.IGNORECASE)
PDF_WHITESPACE_RE = re.compile(r"\s+")


def extract_pdf_text_with_pdftotext(pdf_path: str | Path, *, timeout: int = 120) -> str:
    path = str(pdf_path)
    proc = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", "-nopgbrk", path, "-"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"pdftotext failed for {path}: {stderr or 'unknown error'}")
    return proc.stdout or ""


def _normalize_pdf_line(line: str) -> str:
    return PDF_WHITESPACE_RE.sub(" ", str(line or "").strip())


def _looks_like_pdf_heading(line: str) -> Tuple[bool, int, str]:
    text = _normalize_pdf_line(line)
    if not text:
        return False, 0, ""
    if len(text) > 120:
        return False, 0, ""
    if "@" in text:
        return False, 0, ""
    lowered = text.lower()
    numbered = PDF_HEADING_NUMBERED_RE.match(text)
    if numbered:
        level = min(numbered.group(1).count(".") + 1, 4)
        title = numbered.group(2).strip()
        if 2 <= len(title) <= 100:
            return True, level, title
    if PDF_HEADING_APPENDIX_RE.match(text):
        return True, 1, text
    if sum(ch.isdigit() for ch in text) >= 4 and ("," in text or "†" in text or "*" in text):
        return False, 0, ""
    words = text.split()
    if not words or len(words) > 14:
        return False, 0, ""
    if text.endswith("."):
        return False, 0, ""
    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars < max(3, len(text) // 3):
        return False, 0, ""
    title_like = text.istitle() or text.isupper()
    keyword_like = lowered in {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "approach",
        "framework",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "limitations",
        "limitation",
        "future work",
        "conclusion",
        "conclusions",
        "references",
    }
    if title_like or keyword_like:
        return True, 1, text
    return False, 0, ""


def pdf_text_to_sections(text: str) -> List[Dict]:
    lines = [_normalize_pdf_line(line) for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    sections: List[Dict] = []
    current_title = "Body"
    current_level = 1
    current_lines: List[str] = []
    section_idx = 0

    def flush() -> None:
        nonlocal section_idx, current_lines
        body = safe_text(" ".join(current_lines), limit=4000)
        if not body:
            current_lines = []
            return
        section_idx += 1
        sections.append(
            {
                "section_id": f"pdf::sec_{section_idx}",
                "title": current_title,
                "level": current_level,
                "text": body,
            }
        )
        current_lines = []

    for line in lines:
        is_heading, level, heading_title = _looks_like_pdf_heading(line)
        if is_heading:
            flush()
            current_title = safe_text(heading_title, limit=200) or "Section"
            current_level = level or 1
            continue
        current_lines.append(line)
    flush()

    if not sections:
        body = safe_text(" ".join(lines), limit=4000)
        if body:
            sections.append(
                {
                    "section_id": "pdf::body",
                    "title": "Body",
                    "level": 1,
                    "text": body,
                }
            )
    return sections


def build_content_row_from_source_blob(
    *,
    paper_id: str,
    source_paper_id: str,
    title: str,
    abstract: str,
    blob: bytes,
) -> Dict:
    files = extract_archive_files(blob)
    main_name, main_text = choose_main_tex_file(files)
    expanded = inline_tex_inputs(main_text, files)
    sections = tex_to_sections(main_name, expanded)
    return {
        "paper_id": paper_id,
        "source_paper_id": source_paper_id,
        "title": title,
        "abstract": abstract,
        "source_type": "tex_source",
        "source_url": arxiv_source_url(source_paper_id),
        "main_file": main_name,
        "section_count": len(sections),
        "sections": sections,
    }


def build_content_row_from_html(
    *,
    paper_id: str,
    source_paper_id: str,
    title: str,
    abstract: str,
    html: str,
) -> Dict:
    sections = html_to_sections(html)
    return {
        "paper_id": paper_id,
        "source_paper_id": source_paper_id,
        "title": title,
        "abstract": abstract,
        "source_type": "html",
        "source_url": arxiv_html_url(source_paper_id),
        "section_count": len(sections),
        "sections": sections,
    }


def build_fallback_content_row(
    *,
    paper_id: str,
    source_paper_id: str,
    title: str,
    abstract: str,
) -> Dict:
    sections = fallback_abstract_sections(abstract)
    return {
        "paper_id": paper_id,
        "source_paper_id": source_paper_id,
        "title": title,
        "abstract": abstract,
        "source_type": "abstract_only",
        "source_url": "",
        "section_count": len(sections),
        "sections": sections,
    }


def build_content_row_from_pdf_text(
    *,
    paper_id: str,
    source_paper_id: str,
    title: str,
    abstract: str,
    pdf_text: str,
    source_url: str = "",
) -> Dict:
    sections = pdf_text_to_sections(pdf_text)
    if not sections and abstract:
        sections = fallback_abstract_sections(abstract)
    return {
        "paper_id": paper_id,
        "source_paper_id": source_paper_id,
        "title": title,
        "abstract": abstract,
        "source_type": "pdf_text",
        "source_url": source_url or arxiv_pdf_url(source_paper_id),
        "section_count": len(sections),
        "sections": sections,
    }


def iter_section_texts(row: Dict) -> Iterable[str]:
    for section in row.get("sections") or []:
        text = section.get("text")
        if isinstance(text, str) and text.strip():
            yield text.strip()
