from __future__ import annotations

import csv
import hashlib
import json
import mailbox
import re
from dataclasses import asdict, dataclass, field
from email import policy
from email.parser import BytesParser
from html import unescape
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = {"HAM", "SPAM", "PHISHING", "UNKNOWN"}

LINK_RE = re.compile(r"(https?://|www\.|[A-Za-z0-9.-]+\.[A-Za-z]{2,}/\S*)", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{6,}\d)(?!\d)")
HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
HTML_RE = re.compile(r"(?is)<(html|body|a|div|table|br|p|span)\b")

CATEGORY_PATTERNS = {
    "business": re.compile(r"\b(meeting|proposal|quote|contract|invoice|client|vendor|partner|business|deal|project)\b", re.I),
    "recruiting": re.compile(r"\b(job|role|position|resume|cv|candidate|recruit|hiring|interview|salary)\b", re.I),
    "marketing_newsletter": re.compile(r"\b(newsletter|subscribe|unsubscribe|promotion|coupon|offer|sale|discount|campaign)\b", re.I),
    "transactional": re.compile(r"\b(order|receipt|confirmation|confirmed|booking|payment|transaction|account statement)\b", re.I),
    "delivery_bounce": re.compile(r"\b(delivery|undeliver|bounce|returned mail|mailer-daemon|failed|shipment|tracking)\b", re.I),
    "system_admin": re.compile(r"\b(password|login|quota|mailbox|system|admin|verification|security alert|reset)\b", re.I),
    "finance": re.compile(r"\b(bank|wire|payment|invoice|tax|credit|debit|loan|investment|statement)\b", re.I),
}


@dataclass(slots=True)
class UnifiedRecord:
    text: str
    label: str
    source: str
    split_origin: str = "raw"
    language: str = "unknown"
    sample_origin: str = "real"
    allow_training: bool = False
    allow_evaluation: bool = True
    license_status: str = "unknown"
    contains_link: bool = False
    contains_phone: bool = False
    message_length: int = 0
    category_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["label"] = normalize_label(row["label"])
        return row


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw or "")
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return normalize_text(unescape(raw))


def decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1255", "iso-8859-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_email_bytes(raw: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        decoded = decode_bytes(raw)
        return strip_html(decoded) if HTML_RE.search(decoded) else normalize_text(decoded)

    parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", "")).lower()
            if "attachment" in disposition:
                continue
            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue
            try:
                content = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True)
                content = decode_bytes(payload or b"")
            parts.append(str(content or ""))
    else:
        try:
            parts.append(str(msg.get_content() or ""))
        except Exception:
            payload = msg.get_payload(decode=True)
            parts.append(decode_bytes(payload or raw))

    text = "\n".join(part for part in parts if part)
    return strip_html(text) if HTML_RE.search(text) or msg.get_content_type() == "text/html" else normalize_text(text)


def iter_mbox_texts(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    box = mailbox.mbox(path)
    for idx, message in enumerate(box):
        raw = message.as_bytes(policy=policy.default)
        yield parse_email_bytes(raw), {"mbox_index": idx, "raw_path": str(path)}


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().upper()
    if label in {"0", "HAM", "LEGIT", "LEGITIMATE", "NOT_SPAM", "NOT SPAM", "BENIGN"}:
        return "HAM"
    if label in {"1", "SPAM", "JUNK"}:
        return "SPAM"
    if label in {"2", "PHISH", "PHISHING", "SMISHING"}:
        return "PHISHING"
    return "UNKNOWN"


def infer_label_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    lowered = "/".join(parts)
    if any(part in {"ham", "easy_ham", "easy_ham_2", "hard_ham", "legitimate"} for part in parts):
        return "HAM"
    if any(part in {"spam", "spam_2", "junk"} for part in parts):
        return "SPAM"
    if "phish" in lowered or "smish" in lowered:
        return "PHISHING"
    return "UNKNOWN"


def detect_language(text: str) -> str:
    has_hebrew = bool(HEBREW_RE.search(text or ""))
    has_latin = bool(LATIN_RE.search(text or ""))
    if has_hebrew and has_latin:
        return "hebrew_multilingual"
    if has_hebrew:
        return "hebrew"
    if has_latin:
        return "english_or_latin"
    return "multilingual_or_other"


def category_tags(text: str) -> list[str]:
    tags = [name for name, pattern in CATEGORY_PATTERNS.items() if pattern.search(text or "")]
    if LINK_RE.search(text or ""):
        tags.append("links")
    if PHONE_RE.search(text or ""):
        tags.append("phone_numbers")
    word_count = len((text or "").split())
    if word_count <= 20:
        tags.append("short_message")
    if word_count >= 800:
        tags.append("very_long_message")
    return sorted(set(tags))


def enrich_record(
    *,
    text: str,
    label: str,
    source: str,
    split_origin: str = "raw",
    sample_origin: str = "real",
    allow_training: bool,
    allow_evaluation: bool,
    license_status: str,
    metadata: dict[str, Any] | None = None,
) -> UnifiedRecord:
    clean_text = normalize_text(text)
    return UnifiedRecord(
        text=clean_text,
        label=normalize_label(label),
        source=source,
        split_origin=split_origin,
        language=detect_language(clean_text),
        sample_origin=sample_origin,
        allow_training=allow_training,
        allow_evaluation=allow_evaluation,
        license_status=license_status,
        contains_link=bool(LINK_RE.search(clean_text)),
        contains_phone=bool(PHONE_RE.search(clean_text)),
        message_length=len(clean_text.split()),
        category_tags=category_tags(clean_text),
        metadata=metadata or {},
    )


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8", errors="ignore")).hexdigest()


def template_fingerprint(text: str) -> str:
    value = normalize_text(text).lower()
    value = LINK_RE.sub(" URL ", value)
    value = PHONE_RE.sub(" PHONE ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)*\b", " NUM ", value)
    value = re.sub(r"[a-f0-9]{16,}", " HEX ", value)
    value = re.sub(r"\s+", " ", value)
    return value[:2000].strip()


def template_hash(text: str) -> str:
    return hashlib.sha256(template_fingerprint(text).encode("utf-8", errors="ignore")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean_row = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(clean_row)
