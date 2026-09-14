from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = ROOT / "tmp" / "origin"
DATASET = ROOT / "dataset"


def normalize_cell(value: str) -> str:
    value = html.unescape(value.strip())
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = value.replace("\\|", "|")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    return value.strip()


def split_table_row(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells: list[str] = []
    start = 0
    escaped = False
    for index, char in enumerate(text):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "|" and not escaped:
            cells.append(text[start:index])
            start = index + 1
        escaped = False
    cells.append(text[start:])
    return [normalize_cell(cell) for cell in cells]


def parse_table_qa(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = split_table_row(line)
        if len(cells) < 4:
            continue
        if cells[0] in {"分类", "分类 ", ""} and cells[1] in {"问题", ""}:
            continue
        if all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        question, answer = cells[1], cells[3]
        if question and answer:
            rows.append({"question": question, "answer": answer})
    return rows


FAQ_MARKER = re.compile(
    r"^\s*(?:#{1,6}\s*)?Q\s*\d+\s*[.．:：、]\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)


def parse_faq(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_question: str | None = None
    answer_lines: list[str] = []

    def flush() -> None:
        nonlocal current_question, answer_lines
        if current_question is not None:
            answer = normalize_cell("\n".join(answer_lines))
            if answer:
                records.append({"question": current_question, "answer": answer})
        current_question = None
        answer_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = FAQ_MARKER.match(line)
        if match:
            flush()
            current_question = normalize_cell(match.group(1))
        elif current_question is not None:
            answer_lines.append(line)
    flush()
    return records


def main() -> None:
    DATASET.mkdir(exist_ok=True)
    faq_files = {"IT问答梳理.md", "IT日常问题FAQ.md"}
    metadata_files = {"akasha-metadata.json"}

    corpus: list[dict[str, str]] = []
    skipped: list[str] = []
    for path in sorted(ORIGIN.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.name in faq_files or path.name in metadata_files:
            if path.name in metadata_files:
                skipped.append(path.name)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append(path.name)
            continue
        corpus.append(
            {
                "id": f"doc_{len(corpus) + 1:03d}",
                "title": path.stem,
                "text": text,
            }
        )

    qa_rows = parse_table_qa(ORIGIN / "IT问答梳理.md")
    qa_rows += parse_faq(ORIGIN / "IT日常问题FAQ.md")
    unique_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in qa_rows:
        key = (row["question"], row["answer"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    qa = [
        {"id": f"qa_{index:03d}", **row}
        for index, row in enumerate(unique_rows, start=1)
    ]

    (DATASET / "itfaq_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (DATASET / "itfaq.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"corpus": len(corpus), "qa": len(qa), "skipped": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
