from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import (
    compute_content_hash,
    load_source_index,
    write_source_index,
)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
DEFAULT_CREDENTIALS_PATH = "credentials.json"
DEFAULT_TOKEN_PATH = ".streamlit/gmail_token.json"
DEFAULT_DESTINATION_DIR = "test/mails/gmail"


def _google_modules() -> dict[str, Any]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Missing Gmail OAuth dependencies. Install them with "
            "`pip install -r requirements-demo.txt`."
        ) from exc

    return {
        "Request": Request,
        "Credentials": Credentials,
        "InstalledAppFlow": InstalledAppFlow,
        "build": build,
    }


def _ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_credentials_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Invalid credentials.json: expected a JSON object.")

    client_section = payload.get("installed") or payload.get("web")
    if not isinstance(client_section, dict):
        raise ValueError(
            "Invalid credentials.json: missing `installed` OAuth client configuration."
        )

    required_keys = {"client_id", "auth_uri", "token_uri"}
    missing = sorted(key for key in required_keys if not client_section.get(key))
    if missing:
        raise ValueError(
            "Invalid credentials.json: missing required keys: " + ", ".join(missing)
        )


def _slugify_filename(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "mail"


def _decode_raw_message(raw_value: str) -> bytes:
    padding = "=" * (-len(raw_value) % 4)
    return base64.urlsafe_b64decode(raw_value + padding)


def _build_destination_name(subject: str, message_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    subject_slug = _slugify_filename(subject)
    return f"gmail_{timestamp}_{subject_slug}_{message_id[:12]}.eml"


def _save_credentials(credentials: Any, token_path: str | Path) -> Path:
    token_file = _ensure_parent_dir(token_path)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    return token_file


def save_gmail_credentials_file(
    raw_bytes: bytes,
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
) -> Path:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid credentials.json: unable to parse JSON.") from exc

    _validate_credentials_payload(payload)
    credentials_file = _ensure_parent_dir(credentials_path)
    credentials_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return credentials_file


def remove_gmail_credentials_file(
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
) -> None:
    credentials_file = Path(credentials_path)
    if credentials_file.exists():
        credentials_file.unlink()


def _load_token(credentials_cls: Any, token_path: str | Path) -> Any | None:
    token_file = Path(token_path)
    if not token_file.exists():
        return None
    return credentials_cls.from_authorized_user_file(str(token_file), SCOPES)


def get_gmail_credentials(
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
    allow_browser: bool = True,
) -> Any:
    google = _google_modules()
    credentials_file = Path(credentials_path)
    if not credentials_file.exists():
        raise FileNotFoundError(
            f"Gmail OAuth credentials file not found: {credentials_file.resolve()}"
        )

    creds = _load_token(google["Credentials"], token_path)
    if creds and getattr(creds, "valid", False):
        return creds

    if creds and getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
        try:
            creds.refresh(google["Request"]())
            _save_credentials(creds, token_path)
            return creds
        except Exception:
            creds = None

    if not allow_browser:
        raise RuntimeError(
            "Gmail is not connected. Use the Connect Gmail action to complete OAuth login."
        )

    flow = google["InstalledAppFlow"].from_client_secrets_file(str(credentials_file), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_credentials(creds, token_path)
    return creds


def build_gmail_service(
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
    allow_browser: bool = True,
) -> Any:
    google = _google_modules()
    credentials = get_gmail_credentials(
        credentials_path=credentials_path,
        token_path=token_path,
        allow_browser=allow_browser,
    )
    return google["build"]("gmail", "v1", credentials=credentials, cache_discovery=False)


def get_gmail_profile(
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
    allow_browser: bool = True,
) -> dict[str, Any]:
    service = build_gmail_service(
        credentials_path=credentials_path,
        token_path=token_path,
        allow_browser=allow_browser,
    )
    return service.users().getProfile(userId="me").execute()


def connect_gmail(
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
) -> dict[str, Any]:
    profile = get_gmail_profile(
        credentials_path=credentials_path,
        token_path=token_path,
        allow_browser=True,
    )
    return {
        "connected": True,
        "email_address": str(profile.get("emailAddress", "") or ""),
        "messages_total": int(profile.get("messagesTotal", 0) or 0),
        "threads_total": int(profile.get("threadsTotal", 0) or 0),
    }


def disconnect_gmail(token_path: str | Path = DEFAULT_TOKEN_PATH) -> None:
    token_file = Path(token_path)
    if token_file.exists():
        token_file.unlink()


def import_from_gmail(
    max_messages: int = 20,
    label_ids: tuple[str, ...] = ("INBOX",),
    destination_dir: str | Path = DEFAULT_DESTINATION_DIR,
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
) -> list[Path]:
    service = build_gmail_service(
        credentials_path=credentials_path,
        token_path=token_path,
        allow_browser=True,
    )
    profile = service.users().getProfile(userId="me").execute()
    account_email = str(profile.get("emailAddress", "") or "")

    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    source_index = load_source_index(destination_path)
    source_index["account_email"] = account_email
    source_index["provider"] = "gmail"

    existing_messages = source_index.get("messages", {})
    if not isinstance(existing_messages, dict):
        existing_messages = {}

    # Keep only entries whose snapshot file still exists.
    existing_messages = {
        file_name: metadata
        for file_name, metadata in existing_messages.items()
        if (destination_path / file_name).exists()
    }
    existing_remote_ids = {
        str(metadata.get("remote_message_id", "") or "")
        for metadata in existing_messages.values()
        if metadata.get("remote_message_id")
    }
    existing_hashes = {
        str(metadata.get("content_hash", "") or "")
        for metadata in existing_messages.values()
        if metadata.get("content_hash")
    }

    response = service.users().messages().list(
        userId="me",
        labelIds=list(label_ids),
        maxResults=max_messages,
    ).execute()
    messages = response.get("messages", []) or []
    imported_paths: list[Path] = []
    synced_at = datetime.now(timezone.utc).isoformat()

    for message_stub in messages:
        remote_message_id = str(message_stub.get("id", "") or "")
        if not remote_message_id or remote_message_id in existing_remote_ids:
            continue

        message = service.users().messages().get(
            userId="me",
            id=remote_message_id,
            format="raw",
        ).execute()
        raw_value = message.get("raw")
        if not raw_value:
            continue

        raw_bytes = _decode_raw_message(str(raw_value))
        content_hash = compute_content_hash(raw_bytes)
        if content_hash in existing_hashes:
            continue

        payload_headers = message.get("payload", {}).get("headers", []) or []
        subject = next(
            (
                header.get("value", "")
                for header in payload_headers
                if str(header.get("name", "")).lower() == "subject"
            ),
            "",
        )
        destination_name = _build_destination_name(subject, remote_message_id)
        destination_file = destination_path / destination_name
        suffix = 1
        while destination_file.exists():
            destination_file = destination_path / (
                f"{destination_file.stem}_{suffix}{destination_file.suffix}"
            )
            suffix += 1

        destination_file.write_bytes(raw_bytes)
        imported_paths.append(destination_file)
        existing_remote_ids.add(remote_message_id)
        existing_hashes.add(content_hash)
        existing_messages[destination_file.name] = {
            "account_email": account_email,
            "content_hash": content_hash,
            "provider": "gmail",
            "remote_message_id": remote_message_id,
            "synced_at": synced_at,
            "thread_id": str(message.get("threadId", "") or ""),
        }

    source_index["last_sync_at"] = synced_at
    source_index["messages"] = existing_messages
    write_source_index(destination_path, source_index)
    return imported_paths


def get_gmail_connection_status(
    credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
    token_path: str | Path = DEFAULT_TOKEN_PATH,
    destination_dir: str | Path = DEFAULT_DESTINATION_DIR,
) -> dict[str, Any]:
    credentials_file = Path(credentials_path)
    token_file = Path(token_path)
    source_index = load_source_index(destination_dir)
    status = {
        "account_email": str(source_index.get("account_email", "") or ""),
        "connected": False,
        "credentials_exists": credentials_file.exists(),
        "last_import_count": len(source_index.get("messages", {}) or {}),
        "last_sync_at": str(source_index.get("last_sync_at", "") or ""),
        "token_exists": token_file.exists(),
    }

    if not status["credentials_exists"] or not status["token_exists"]:
        return status

    try:
        profile = get_gmail_profile(
            credentials_path=credentials_path,
            token_path=token_path,
            allow_browser=False,
        )
    except Exception:
        return status

    status["account_email"] = str(profile.get("emailAddress", "") or status["account_email"])
    status["connected"] = True
    status["messages_total"] = int(profile.get("messagesTotal", 0) or 0)
    status["threads_total"] = int(profile.get("threadsTotal", 0) or 0)
    return status
