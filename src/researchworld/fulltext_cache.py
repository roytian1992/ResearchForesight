from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from researchworld.content import arxiv_pdf_url, build_content_row_from_pdf_text, extract_pdf_text_with_pdftotext
from researchworld.offline_kb import HybridRetriever, RetrievalDoc, clip_text, normalize_ws, on_or_before


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def has_version_suffix(source_paper_id: str) -> bool:
    text = str(source_paper_id or "").strip().lower()
    return bool(text) and "v" in text and text.rsplit("v", 1)[-1].isdigit()


class LocalFulltextCache:
    def __init__(
        self,
        *,
        domain_id: str,
        papers_jsonl: Path,
        cache_root: Path,
    ):
        self.domain_id = domain_id
        self.papers_jsonl = papers_jsonl
        self.cache_root = cache_root / domain_id
        self.pdf_dir = self.cache_root / "pdfs"
        self.content_path = self.cache_root / "content_pdf.jsonl"
        self.error_path = self.cache_root / "content_pdf.errors.jsonl"
        self.manifest_path = self.cache_root / "download_manifest.jsonl"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.papers_by_id: Dict[str, Dict[str, Any]] = {}
        for row in _iter_jsonl(self.papers_jsonl) or []:
            paper_id = str(row.get("paper_id") or "").strip()
            if paper_id:
                self.papers_by_id[paper_id] = row

        self.content_by_paper_id: Dict[str, Dict[str, Any]] = {}
        for row in _iter_jsonl(self.content_path) or []:
            paper_id = str(row.get("paper_id") or "").strip()
            if paper_id:
                self.content_by_paper_id[paper_id] = row

    def _append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @contextmanager
    def _paper_lock(self, paper_id: str, *, timeout: int = 300):
        lock_path = self.cache_root / "locks" / f"{paper_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        fd: Optional[int] = None
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                break
            except FileExistsError:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"timed out waiting for lock {lock_path}")
                time.sleep(0.5)
        try:
            yield
        finally:
            try:
                if fd is not None:
                    os.close(fd)
            except Exception:
                pass
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except Exception:
                pass

    def _verbose(self) -> bool:
        return os.environ.get("RTL_VERBOSE_COI", "").strip().lower() in {"1", "true", "yes", "y"}

    def _log(self, message: str) -> None:
        if self._verbose():
            print(f"[LocalFulltextCache][debug] {message}", flush=True)

    def _download_pdf(self, *, pdf_url: str, tmp_pdf_path: Path, timeout: int) -> Dict[str, Any]:
        started = time.monotonic()
        aria2 = shutil.which("aria2c")
        errors: List[str] = []

        if aria2:
            aria2_cmd = [
                aria2,
                "--dir",
                str(tmp_pdf_path.parent),
                "--out",
                tmp_pdf_path.name,
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=1M",
                "--continue=true",
                "--timeout=120",
                "--connect-timeout=20",
                "--summary-interval=0",
                "--console-log-level=warn",
                "--file-allocation=none",
                "--user-agent=ResearchTrajectoryLab/0.1",
                pdf_url,
            ]
            proc = subprocess.run(
                aria2_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout + 10,
                check=False,
            )
            if proc.returncode == 0 and tmp_pdf_path.exists() and tmp_pdf_path.stat().st_size > 0:
                elapsed = time.monotonic() - started
                return {
                    "bytes": tmp_pdf_path.stat().st_size,
                    "elapsed_sec": round(elapsed, 3),
                    "downloader": "aria2c",
                }
            errors.append(f"aria2c rc={proc.returncode} stderr={proc.stderr.strip()[:300]} stdout={proc.stdout.strip()[:300]}")

        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("; ".join(errors) if errors else "neither aria2c nor curl is available")
        curl_cmd = [
            curl,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            str(timeout),
            "-A",
            "ResearchTrajectoryLab/0.1",
            "-C",
            "-",
            "-o",
            str(tmp_pdf_path),
            pdf_url,
        ]
        proc = subprocess.run(
            curl_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout + 10,
            check=False,
        )
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            errors.append(f"curl rc={proc.returncode} stderr={proc.stderr.strip()[:400]}")
            raise RuntimeError("; ".join(errors))
        total_bytes = tmp_pdf_path.stat().st_size if tmp_pdf_path.exists() else 0
        if total_bytes <= 0:
            raise RuntimeError("empty file after curl download")
        return {
            "bytes": total_bytes,
            "elapsed_sec": round(elapsed, 3),
            "downloader": "curl",
        }

    def get_content(self, paper_id: str, *, cutoff_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        row = self.content_by_paper_id.get(str(paper_id))
        if not row:
            return None
        paper = self.papers_by_id.get(str(paper_id), {})
        published = paper.get("published_date") or paper.get("published")
        if not on_or_before(published, cutoff_date):
            return None
        return row

    def _reload_content_row(self, paper_id: str) -> Optional[Dict[str, Any]]:
        paper_id = str(paper_id)
        if not self.content_path.exists():
            return None
        found = None
        for row in _iter_jsonl(self.content_path) or []:
            if str(row.get("paper_id") or "") == paper_id:
                found = row
        if found is not None:
            self.content_by_paper_id[paper_id] = found
        return found

    def resolve_pdf_path(self, paper_id: str) -> Optional[Path]:
        paper = self.papers_by_id.get(str(paper_id)) or {}
        source_paper_id = str(paper.get("source_paper_id") or "").strip()
        candidates = []
        if source_paper_id:
            candidates.append(self.pdf_dir / f"{source_paper_id}.pdf")
        candidates.append(self.pdf_dir / f"{paper_id}.pdf")
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def ensure_content(self, paper_id: str, *, allow_fetch: bool = False, timeout: int = 120) -> Optional[Dict[str, Any]]:
        cached = self.get_content(paper_id)
        if cached is not None:
            return cached
        if not allow_fetch:
            return None

        paper = self.papers_by_id.get(str(paper_id)) or {}
        if not paper:
            return None
        source_paper_id = str(paper.get("source_paper_id") or "").strip()
        if not has_version_suffix(source_paper_id):
            self._append_jsonl(
                self.error_path,
                {
                    "paper_id": paper_id,
                    "source_paper_id": source_paper_id,
                    "error": "source_paper_id must include explicit version suffix",
                },
            )
            return None

        with self._paper_lock(str(paper_id), timeout=max(timeout * 3, 300)):
            cached = self.get_content(paper_id)
            if cached is not None:
                return cached
            cached = self._reload_content_row(paper_id)
            if cached is not None:
                return cached

            pdf_path = self.resolve_pdf_path(paper_id)
            if pdf_path is None:
                pdf_url = str(paper.get("pdf_url") or arxiv_pdf_url(source_paper_id))
                pdf_path = self.pdf_dir / f"{source_paper_id}.pdf"
                tmp_pdf_path = self.pdf_dir / f"{source_paper_id}.pdf.part"
                try:
                    self._log(f"fetch_start paper_id={paper_id} source_paper_id={source_paper_id} url={pdf_url}")
                    stats = self._download_pdf(pdf_url=pdf_url, tmp_pdf_path=tmp_pdf_path, timeout=timeout)
                    total_bytes = int(stats["bytes"])
                    if total_bytes <= 0:
                        raise RuntimeError("empty file after download")
                    with tmp_pdf_path.open("rb") as check_handle:
                        prefix = check_handle.read(8)
                    if not prefix.startswith(b"%PDF-"):
                        raise RuntimeError("downloaded file is not a valid PDF prefix")
                    os.replace(tmp_pdf_path, pdf_path)
                    self._append_jsonl(
                        self.manifest_path,
                        {
                            "paper_id": paper_id,
                            "source_paper_id": source_paper_id,
                            "pdf_url": pdf_url,
                            "pdf_path": str(pdf_path),
                            "bytes": total_bytes,
                            "elapsed_sec": stats["elapsed_sec"],
                            "downloader": stats.get("downloader"),
                            "status": "downloaded",
                        },
                    )
                    self._log(
                        f"fetch_done paper_id={paper_id} source_paper_id={source_paper_id} bytes={total_bytes} "
                        f"elapsed_sec={stats['elapsed_sec']} downloader={stats.get('downloader')}"
                    )
                except Exception as exc:
                    try:
                        if tmp_pdf_path.exists():
                            aria2_state = tmp_pdf_path.with_name(tmp_pdf_path.name + ".aria2")
                            if not aria2_state.exists():
                                tmp_pdf_path.unlink()
                    except Exception:
                        pass
                    self._append_jsonl(
                        self.error_path,
                        {
                            "paper_id": paper_id,
                            "source_paper_id": source_paper_id,
                            "error": f"pdf download failed: {exc}",
                        },
                    )
                    return None

            try:
                pdf_text = extract_pdf_text_with_pdftotext(pdf_path, timeout=timeout)
                row = build_content_row_from_pdf_text(
                    paper_id=str(paper.get("paper_id") or paper_id),
                    source_paper_id=source_paper_id,
                    title=str(paper.get("title") or ""),
                    abstract=str(paper.get("abstract") or ""),
                    pdf_text=pdf_text,
                    source_url=str(paper.get("pdf_url") or arxiv_pdf_url(source_paper_id)),
                )
                row["pdf_path"] = str(pdf_path)
                row["pdf_char_count"] = len(pdf_text)
                self.content_by_paper_id[str(paper_id)] = row
                self._append_jsonl(self.content_path, row)
                return row
            except Exception as exc:
                self._append_jsonl(
                    self.error_path,
                    {
                        "paper_id": paper_id,
                        "source_paper_id": source_paper_id,
                        "pdf_path": str(pdf_path),
                        "error": f"pdf parse failed: {exc}",
                    },
                )
                return None

    def build_section_retriever(
        self,
        *,
        paper_ids: Iterable[str],
        cutoff_date: Optional[str] = None,
        allow_fetch: bool = False,
    ) -> HybridRetriever:
        docs: List[RetrievalDoc] = []
        for paper_id in {str(x) for x in paper_ids if str(x).strip()}:
            row = self.ensure_content(paper_id, allow_fetch=allow_fetch)
            if row is None:
                continue
            paper = self.papers_by_id.get(paper_id) or {}
            published = paper.get("published_date") or paper.get("published")
            if not on_or_before(published, cutoff_date):
                continue
            for section in row.get("sections") or []:
                text = clip_text(section.get("text") or "", 2200)
                if not normalize_ws(text):
                    continue
                docs.append(
                    RetrievalDoc(
                        doc_id=f"fulltext::{paper_id}::{section.get('section_id')}",
                        paper_id=paper_id,
                        title=f"{row.get('title') or ''} / {section.get('title') or ''}",
                        text="\n".join(
                            part
                            for part in [
                                f"Paper: {row.get('title') or ''}",
                                f"Section: {section.get('title') or ''}",
                                f"Level: {section.get('level')}",
                                text,
                            ]
                            if normalize_ws(part)
                        ),
                        meta={
                            "paper_title": row.get("title"),
                            "section_title": section.get("title"),
                            "section_id": section.get("section_id"),
                            "level": section.get("level"),
                            "source_type": row.get("source_type"),
                            "pdf_path": row.get("pdf_path"),
                        },
                    )
                )
        return HybridRetriever(docs)
