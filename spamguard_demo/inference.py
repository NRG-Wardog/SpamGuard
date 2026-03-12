from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from html import unescape
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = "globalSpm.model"
DEFAULT_MAX_LEN = 256
SOURCE_INDEX_FILENAME = ".spamguard_source_index.json"
SOURCE_INDEX_VERSION = 1
CHARSET_ALIASES = {
    "iso-8859-8-e": "iso-8859-8",
    "iso-8859-8-i": "iso-8859-8",
}
RESULT_FIELDNAMES = [
    "source",
    "provider",
    "original_path",
    "stored_result_path",
    "prediction",
    "spam_probability",
    "message_id",
    "remote_message_id",
    "thread_id",
    "subject",
    "from",
    "synced_at",
    "processed_at",
]

HEADER_ALIASES = {
    "from": ("from", "מאת"),
    "to": ("to", "אל"),
    "subject": ("subject", "נושא"),
    "message_id": ("message-id", "message id", "מזהה הודעה"),
}

_MODEL_CACHE: dict[tuple[str, int, str], "ModelBundle"] = {}


@dataclass(slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    max_len: int
    model_dir: Path


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return _normalize_text(unescape(raw))


def _decode_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1255", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _normalize_charset_name(charset: str | None) -> str:
    if not charset:
        return "utf-8"
    normalized = charset.strip().strip('"').strip("'").lower()
    return CHARSET_ALIASES.get(normalized, normalized or "utf-8")


def _decode_part_bytes(raw_bytes: bytes, charset: str | None) -> str:
    candidates = []
    normalized_charset = _normalize_charset_name(charset)
    if normalized_charset:
        candidates.append(normalized_charset)

    for fallback in ("utf-8", "cp1255", "latin-1"):
        if fallback not in candidates:
            candidates.append(fallback)

    for encoding in candidates:
        try:
            return raw_bytes.decode(encoding, errors="replace")
        except LookupError:
            continue

    return raw_bytes.decode("utf-8", errors="replace")


def _canonical_header_name(label: str) -> str | None:
    normalized = label.strip().lower()
    for canonical_name, aliases in HEADER_ALIASES.items():
        if normalized in aliases:
            return canonical_name
    return None


def _parse_pseudo_headers(raw_text: str) -> dict[str, str]:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    header_block, separator, body = normalized.partition("\n\n")
    lines = header_block.splitlines()

    parsed_headers: dict[str, str] = {}
    consumed_lines = 0

    for line in lines[:16]:
        if ":" not in line:
            break

        label, value = line.split(":", 1)
        canonical_name = _canonical_header_name(label)
        if canonical_name is None:
            if label.strip().lower() in {"mime-version", "content-type", "גרסת mime", "סוג תוכן"}:
                consumed_lines += 1
                continue
            break

        parsed_headers[canonical_name] = value.strip()
        consumed_lines += 1

    if consumed_lines:
        trailing_lines = lines[consumed_lines:]
        remaining_parts = []
        if trailing_lines:
            remaining_parts.append("\n".join(trailing_lines))
        if separator:
            remaining_parts.append(body)
        parsed_headers["body"] = "\n\n".join(part for part in remaining_parts if part).strip()
    else:
        parsed_headers["body"] = body.strip() if separator else normalized.strip()

    return parsed_headers


def _coerce_part_content(part: Any) -> str:
    try:
        content = part.get_content()
    except Exception:
        content = part.get_payload(decode=True)

    if isinstance(content, bytes):
        charset = part.get_content_charset() or "utf-8"
        return _decode_part_bytes(content, charset)
    return str(content or "")


def _extract_body_from_message(msg: Any) -> str:
    body_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", "")).lower()
            if "attachment" in disposition:
                continue
            if part.get_content_type() in {"text/plain", "text/html"}:
                body_parts.append(_coerce_part_content(part))
    else:
        body_parts.append(_coerce_part_content(msg))

    return _strip_html("\n".join(part for part in body_parts if part))


def _preview_text(text: str, limit: int = 220) -> str:
    compact = _normalize_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _safe_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _slugify_filename(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "mail"


def compute_content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def extract_eml_metadata(raw_bytes: bytes) -> dict[str, str]:
    raw_text = _normalize_text(_decode_bytes(raw_bytes))
    pseudo_headers = _parse_pseudo_headers(raw_text)
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    subject = str(message.get("subject", "") or "").strip()
    from_value = str(message.get("from", "") or "").strip()
    to_value = str(message.get("to", "") or "").strip()
    message_id = str(message.get("message-id", "") or "").strip()
    body = _extract_body_from_message(message)

    if not any((subject, from_value, to_value)) and any(
        pseudo_headers.get(key) for key in ("subject", "from", "to")
    ):
        subject = pseudo_headers.get("subject", "")
        from_value = pseudo_headers.get("from", "")
        to_value = pseudo_headers.get("to", "")
        message_id = pseudo_headers.get("message_id", "")
        body = pseudo_headers.get("body", "")
    else:
        subject = subject or pseudo_headers.get("subject", "")
        from_value = from_value or pseudo_headers.get("from", "")
        to_value = to_value or pseudo_headers.get("to", "")
        message_id = message_id or pseudo_headers.get("message_id", "")
        body = body or pseudo_headers.get("body", "")

    body = _strip_html(body or raw_text)
    combined_text = _normalize_text(
        f"From: {from_value}\nTo: {to_value}\nSubject: {subject}\n\n{body}"
    )

    return {
        "body": body,
        "from": from_value,
        "message_id": message_id,
        "subject": subject,
        "text": combined_text,
        "text_preview": _preview_text(body or combined_text),
        "to": to_value,
    }


def parse_eml_bytes(raw_bytes: bytes) -> str:
    return extract_eml_metadata(raw_bytes)["text"]


def compute_message_identity(raw_bytes: bytes) -> str:
    metadata = extract_eml_metadata(raw_bytes)
    if metadata["message_id"]:
        return f"message-id:{metadata['message_id']}"
    return f"sha256:{compute_content_hash(raw_bytes)}"


def ensure_demo_dirs(base_dir: str | Path = "test") -> dict[str, Path]:
    base_path = Path(base_dir)
    dirs = {
        "base": base_path,
        "mails_root": base_path / "mails",
        "local_mail": base_path / "mails" / "local",
        "gmail_mail": base_path / "mails" / "gmail",
        "result_root": base_path / "result",
        "spam_result": base_path / "result" / "spam",
        "ham_result": base_path / "result" / "ham",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def _default_source_index(provider: str = "local") -> dict[str, Any]:
    return {
        "account_email": "",
        "last_sync_at": "",
        "messages": {},
        "provider": provider,
        "version": SOURCE_INDEX_VERSION,
    }


def load_source_index(source_dir: str | Path) -> dict[str, Any]:
    source_path = Path(source_dir)
    provider = "gmail" if source_path.name.lower() == "gmail" else "local"
    index_path = source_path / SOURCE_INDEX_FILENAME
    if not index_path.exists():
        return _default_source_index(provider=provider)

    with index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        return _default_source_index(provider=provider)

    messages = data.get("messages", {})
    if not isinstance(messages, dict):
        messages = {}

    normalized = _default_source_index(provider=str(data.get("provider", provider)))
    normalized["account_email"] = str(data.get("account_email", "") or "")
    normalized["last_sync_at"] = str(data.get("last_sync_at", "") or "")
    normalized["messages"] = messages
    normalized["version"] = int(data.get("version", SOURCE_INDEX_VERSION) or SOURCE_INDEX_VERSION)
    return normalized


def write_source_index(source_dir: str | Path, data: Mapping[str, Any]) -> Path:
    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)
    index_path = source_path / SOURCE_INDEX_FILENAME
    payload = {
        "account_email": str(data.get("account_email", "") or ""),
        "last_sync_at": str(data.get("last_sync_at", "") or ""),
        "messages": dict(data.get("messages", {}) or {}),
        "provider": str(data.get("provider", source_path.name.lower() or "local")),
        "version": int(data.get("version", SOURCE_INDEX_VERSION) or SOURCE_INDEX_VERSION),
    }
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    return index_path


def seed_demo_mailbox(base_dir: str | Path = "test", sample_dir: str | Path | None = None) -> list[Path]:
    dirs = ensure_demo_dirs(base_dir)
    destination_dir = dirs["local_mail"]
    source_dir = Path(sample_dir) if sample_dir is not None else dirs["base"]

    if any(destination_dir.glob("*.eml")):
        return []

    copied_paths: list[Path] = []
    for sample_path in sorted(source_dir.glob("*.eml")):
        destination_path = destination_dir / sample_path.name
        shutil.copy2(sample_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def load_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    max_len: int = DEFAULT_MAX_LEN,
    device: str | None = None,
) -> ModelBundle:
    model_path = Path(model_dir).resolve()
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (str(model_path), max_len, device_name)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    torch_device = torch.device(device_name)
    model.to(torch_device)
    model.eval()

    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        max_len=max_len,
        model_dir=model_path,
    )
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def predict_text(
    text: str,
    threshold: float = 0.5,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    max_len: int = DEFAULT_MAX_LEN,
) -> dict[str, float | str]:
    bundle = load_model(model_dir=model_dir, max_len=max_len)
    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=bundle.max_len,
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = bundle.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    spam_probability = float(probs[1])
    return {
        "prediction": "SPAM" if spam_probability >= threshold else "HAM",
        "spam_probability": spam_probability,
    }


def predict_eml_file(
    eml_path: str | Path,
    threshold: float = 0.5,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    max_len: int = DEFAULT_MAX_LEN,
) -> dict[str, Any]:
    path = Path(eml_path)
    if not path.exists():
        raise FileNotFoundError(f"EML file not found: {path}")

    raw_bytes = path.read_bytes()
    metadata = extract_eml_metadata(raw_bytes)
    prediction = predict_text(
        metadata["text"],
        threshold=threshold,
        model_dir=model_dir,
        max_len=max_len,
    )

    return {
        "body": metadata["body"],
        "content_hash": compute_content_hash(raw_bytes),
        "from": metadata["from"],
        "message_id": metadata["message_id"],
        "original_path": _safe_relative_path(path),
        "prediction": prediction["prediction"],
        "spam_probability": prediction["spam_probability"],
        "subject": metadata["subject"],
        "text": metadata["text"],
        "text_preview": metadata["text_preview"],
        "to": metadata["to"],
    }


def iter_eml_files(input_dir: str | Path) -> list[Path]:
    path = Path(input_dir)
    if not path.exists():
        return []
    return sorted(file_path for file_path in path.rglob("*.eml") if file_path.is_file())


def _clear_result_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for file_path in path.glob("*.eml"):
        if file_path.is_file():
            file_path.unlink()


def _write_results_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_path = output_dir / "results.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.get("source", ""),
                    "provider": row.get("provider", ""),
                    "original_path": row.get("original_path", ""),
                    "stored_result_path": row.get("stored_result_path", ""),
                    "prediction": row.get("prediction", ""),
                    "spam_probability": row.get("spam_probability", ""),
                    "message_id": row.get("message_id", ""),
                    "remote_message_id": row.get("remote_message_id", ""),
                    "thread_id": row.get("thread_id", ""),
                    "subject": row.get("subject", ""),
                    "from": row.get("from", ""),
                    "synced_at": row.get("synced_at", ""),
                    "processed_at": row.get("processed_at", ""),
                }
            )
    return manifest_path


def load_results_manifest(output_dir: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(output_dir) / "results.csv"
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def classify_mail_sources(
    input_dirs: Mapping[str, str | Path] | Iterable[str | Path],
    output_dir: str | Path,
    copy_mode: str = "copy",
    threshold: float = 0.5,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    max_len: int = DEFAULT_MAX_LEN,
    reset_output: bool = True,
) -> list[dict[str, Any]]:
    if copy_mode not in {"copy", "move"}:
        raise ValueError("copy_mode must be 'copy' or 'move'")

    if isinstance(input_dirs, Mapping):
        source_map = {source: Path(path) for source, path in input_dirs.items()}
    else:
        source_map = {Path(path).name: Path(path) for path in input_dirs}

    output_root = Path(output_dir)
    spam_dir = output_root / "spam"
    ham_dir = output_root / "ham"
    spam_dir.mkdir(parents=True, exist_ok=True)
    ham_dir.mkdir(parents=True, exist_ok=True)

    if reset_output:
        _clear_result_dir(spam_dir)
        _clear_result_dir(ham_dir)

    results: list[dict[str, Any]] = []

    for source_name, source_dir in source_map.items():
        source_index = load_source_index(source_dir)
        source_messages = source_index.get("messages", {})
        if not isinstance(source_messages, dict):
            source_messages = {}
        provider_default = str(source_index.get("provider", source_name) or source_name)

        for mail_path in iter_eml_files(source_dir):
            source_metadata = source_messages.get(mail_path.name, {})
            if not isinstance(source_metadata, dict):
                source_metadata = {}
            prediction = predict_eml_file(
                mail_path,
                threshold=threshold,
                model_dir=model_dir,
                max_len=max_len,
            )
            destination_root = spam_dir if prediction["prediction"] == "SPAM" else ham_dir
            destination_name = (
                f"{_slugify_filename(source_name)}_"
                f"{_slugify_filename(mail_path.stem)}_"
                f"{prediction['content_hash'][:12]}.eml"
            )
            destination_path = destination_root / destination_name

            if copy_mode == "move":
                shutil.move(str(mail_path), str(destination_path))
            else:
                shutil.copy2(mail_path, destination_path)

            results.append(
                {
                    "source": source_name,
                    "provider": source_metadata.get("provider", provider_default),
                    "original_path": prediction["original_path"],
                    "stored_result_path": _safe_relative_path(destination_path),
                    "prediction": prediction["prediction"],
                    "spam_probability": prediction["spam_probability"],
                    "message_id": prediction["message_id"],
                    "remote_message_id": source_metadata.get("remote_message_id", ""),
                    "thread_id": source_metadata.get("thread_id", ""),
                    "subject": prediction["subject"],
                    "from": prediction["from"],
                    "synced_at": source_metadata.get("synced_at", ""),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    results.sort(key=lambda row: (row["source"], row["original_path"]))
    _write_results_manifest(output_root, results)
    return results


def classify_mail_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    copy_mode: str = "copy",
    threshold: float = 0.5,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    max_len: int = DEFAULT_MAX_LEN,
    source_name: str | None = None,
    reset_output: bool = True,
) -> list[dict[str, Any]]:
    input_path = Path(input_dir)
    source = source_name or input_path.name or "local"
    return classify_mail_sources(
        {source: input_path},
        output_dir=output_dir,
        copy_mode=copy_mode,
        threshold=threshold,
        model_dir=model_dir,
        max_len=max_len,
        reset_output=reset_output,
    )


def _parse_probability(value: str | float | None) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def collect_mail_items(base_dir: str | Path = "test") -> list[dict[str, Any]]:
    dirs = ensure_demo_dirs(base_dir)
    manifest_by_path = {
        row["original_path"]: row for row in load_results_manifest(dirs["result_root"])
    }

    items: list[dict[str, Any]] = []
    source_dirs = {
        "local": dirs["local_mail"],
        "gmail": dirs["gmail_mail"],
    }

    for source_name, source_dir in source_dirs.items():
        source_index = load_source_index(source_dir)
        source_messages = source_index.get("messages", {})
        if not isinstance(source_messages, dict):
            source_messages = {}
        provider_default = str(source_index.get("provider", source_name) or source_name)
        account_email = str(source_index.get("account_email", "") or "")
        for mail_path in iter_eml_files(source_dir):
            raw_bytes = mail_path.read_bytes()
            metadata = extract_eml_metadata(raw_bytes)
            original_path = _safe_relative_path(mail_path)
            manifest_row = manifest_by_path.get(original_path, {})
            source_metadata = source_messages.get(mail_path.name, {})
            if not isinstance(source_metadata, dict):
                source_metadata = {}
            items.append(
                {
                    "account_email": source_metadata.get("account_email", account_email),
                    "body": metadata["body"],
                    "from": metadata["from"],
                    "message_id": metadata["message_id"],
                    "original_path": original_path,
                    "prediction": manifest_row.get("prediction", "UNCLASSIFIED"),
                    "processed_at": manifest_row.get("processed_at", ""),
                    "provider": manifest_row.get("provider", source_metadata.get("provider", provider_default)),
                    "remote_message_id": manifest_row.get(
                        "remote_message_id", source_metadata.get("remote_message_id", "")
                    ),
                    "source": source_name,
                    "spam_probability": _parse_probability(manifest_row.get("spam_probability")),
                    "stored_result_path": manifest_row.get("stored_result_path", ""),
                    "subject": metadata["subject"],
                    "synced_at": manifest_row.get("synced_at", source_metadata.get("synced_at", "")),
                    "text_preview": metadata["text_preview"],
                    "thread_id": manifest_row.get("thread_id", source_metadata.get("thread_id", "")),
                    "to": metadata["to"],
                }
            )

    items.sort(key=lambda row: (row["source"], row["subject"] or row["original_path"]))
    return items
