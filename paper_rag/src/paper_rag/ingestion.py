from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


INVENTORY_FIELDS = [
    "paper_id",
    "source_path",
    "file_name",
    "parent_dir",
    "file_size",
    "modified_time",
    "status",
    "error",
]

PAPER_ID_PATTERN = re.compile(r"^p(\d{6})$")


@dataclass(frozen=True)
class PaperFile:
    paper_id: str
    path: Path
    source_path: str
    file_name: str
    parent_dir: str
    file_size: int
    modified_time: str


@dataclass(frozen=True)
class PageText:
    paper_id: str
    source_path: str
    file_name: str
    page_number: int
    text: str


def normalize_path(path: Path, root: Path | None = None) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def load_existing_paper_ids(inventory_path: Path) -> dict[str, str]:
    if not inventory_path.exists():
        return {}

    with inventory_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {
            row["source_path"]: row["paper_id"]
            for row in reader
            if row.get("source_path") and row.get("paper_id")
        }


def next_paper_id(existing_ids: Iterable[str], next_index: int) -> tuple[str, int]:
    used_numbers = {
        int(match.group(1))
        for paper_id in existing_ids
        if (match := PAPER_ID_PATTERN.match(paper_id))
    }

    while next_index in used_numbers:
        next_index += 1
    return f"p{next_index:06d}", next_index + 1


def max_paper_index(existing_ids: Iterable[str]) -> int:
    numbers = [
        int(match.group(1))
        for paper_id in existing_ids
        if (match := PAPER_ID_PATTERN.match(paper_id))
    ]
    return max(numbers, default=0)


def scan_pdfs(
    input_dir: Path,
    repo_root: Path | None = None,
    existing_paper_ids: dict[str, str] | None = None,
) -> list[PaperFile]:
    if not input_dir.exists():
        return []

    pdf_paths = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: normalize_path(path, repo_root).lower(),
    )

    papers: list[PaperFile] = []
    existing_paper_ids = existing_paper_ids or {}
    next_index = max_paper_index(existing_paper_ids.values()) + 1

    for path in pdf_paths:
        stat = path.stat()
        source_path = normalize_path(path, repo_root)
        paper_id = existing_paper_ids.get(source_path)
        if paper_id is None:
            paper_id, next_index = next_paper_id(existing_paper_ids.values(), next_index)
            existing_paper_ids[source_path] = paper_id

        papers.append(
            PaperFile(
                paper_id=paper_id,
                path=path,
                source_path=source_path,
                file_name=path.name,
                parent_dir=normalize_path(path.parent, repo_root),
                file_size=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            )
        )
    return papers


def extract_pages(pdf_path: Path) -> list[str]:
    try:
        try:
            import pymupdf  # type: ignore
        except ImportError:
            import fitz as pymupdf  # type: ignore

        pages: list[str] = []
        with pymupdf.open(pdf_path) as document:
            for page in document:
                pages.append(page.get_text("text") or "")
        return pages
    except ImportError:
        pass

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires PyMuPDF or pypdf to be installed.") from exc

    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalized = " ".join(sanitize_text(text).split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def iter_page_chunks(
    pages: Iterable[PageText],
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[dict[str, object]]:
    for page in pages:
        for chunk_index, text in enumerate(chunk_text(page.text, chunk_size, chunk_overlap)):
            yield {
                "chunk_id": f"{page.paper_id}_pg{page.page_number:04d}_c{chunk_index:04d}",
                "paper_id": page.paper_id,
                "source_path": page.source_path,
                "file_name": page.file_name,
                "page_number": page.page_number,
                "chunk_index": chunk_index,
                "text": text,
            }


def write_inventory(rows: list[dict[str, object]], inventory_path: Path) -> None:
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_chunks(chunks: Iterable[dict[str, object]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
    return count


def ingest_pdfs(
    input_dir: Path,
    inventory_path: Path,
    output_path: Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    repo_root: Path | None = None,
) -> tuple[int, int, int]:
    existing_paper_ids = load_existing_paper_ids(inventory_path)
    papers = scan_pdfs(input_dir, repo_root=repo_root, existing_paper_ids=existing_paper_ids)
    inventory_rows: list[dict[str, object]] = []
    failed_count = 0
    chunk_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for paper in papers:
            row: dict[str, object] = {
                "paper_id": paper.paper_id,
                "source_path": paper.source_path,
                "file_name": paper.file_name,
                "parent_dir": paper.parent_dir,
                "file_size": paper.file_size,
                "modified_time": paper.modified_time,
                "status": "success",
                "error": "",
            }

            try:
                raw_pages = extract_pages(paper.path)
                page_texts = (
                    PageText(
                        paper_id=paper.paper_id,
                        source_path=paper.source_path,
                        file_name=paper.file_name,
                        page_number=page_number,
                        text=text,
                    )
                    for page_number, text in enumerate(raw_pages, start=1)
                )
                paper_chunks = list(iter_page_chunks(page_texts, chunk_size, chunk_overlap))
                for chunk in paper_chunks:
                    output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1
            except Exception as exc:  # noqa: BLE001 - failures are recorded per PDF.
                failed_count += 1
                row["status"] = "failed"
                row["error"] = str(exc)

            inventory_rows.append(row)

    write_inventory(inventory_rows, inventory_path)
    return len(papers), chunk_count, failed_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan and ingest local research paper PDFs.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing PDF files.")
    parser.add_argument("--inventory", type=Path, required=True, help="Output CSV inventory path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL chunks path.")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Maximum chunk length in characters.")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Character overlap between chunks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = Path.cwd()

    paper_count, chunk_count, failed_count = ingest_pdfs(
        input_dir=args.input_dir,
        inventory_path=args.inventory,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        repo_root=repo_root,
    )

    print(
        f"Ingested {paper_count} PDF(s), wrote {chunk_count} chunk(s), "
        f"recorded {failed_count} failure(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
