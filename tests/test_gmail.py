from __future__ import annotations

import base64
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spamguard_demo.gmail import (
    disconnect_gmail,
    get_gmail_credentials,
    import_from_gmail,
    remove_gmail_credentials_file,
    save_gmail_credentials_file,
)
from spamguard_demo.inference import load_source_index


class FakeCredentials:
    from_authorized_user_file_result = None

    def __init__(self, *, valid: bool, expired: bool, refresh_token: str | None, email: str = "") -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.email = email
        self.refreshed = False
        self.refresh_raises = False

    def refresh(self, _request) -> None:
        if self.refresh_raises:
            raise RuntimeError("refresh failed")
        self.valid = True
        self.expired = False
        self.refreshed = True

    def to_json(self) -> str:
        return json.dumps({"token": "token", "email": self.email})

    @classmethod
    def from_authorized_user_file(cls, _path: str, _scopes: list[str]):
        return cls.from_authorized_user_file_result


class FakeFlow:
    def __init__(self, credentials: FakeCredentials) -> None:
        self.credentials = credentials
        self.run_count = 0

    def run_local_server(self, port: int = 0) -> FakeCredentials:
        self.run_count += 1
        return self.credentials


class FakeInstalledAppFlow:
    next_flow: FakeFlow | None = None

    @classmethod
    def from_client_secrets_file(cls, _path: str, _scopes: list[str]) -> FakeFlow:
        assert cls.next_flow is not None
        return cls.next_flow


class FakeListRequest:
    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages

    def execute(self) -> dict[str, object]:
        return {"messages": self.messages}


class FakeMessageGetRequest:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def execute(self) -> dict[str, object]:
        return self.payload


class FakeProfileRequest:
    def __init__(self, email: str) -> None:
        self.email = email

    def execute(self) -> dict[str, object]:
        return {
            "emailAddress": self.email,
            "messagesTotal": 7,
            "threadsTotal": 5,
        }


class FakeMessagesResource:
    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self.payloads = payloads
        self.list_calls: list[dict[str, object]] = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return FakeListRequest([{"id": key} for key in self.payloads])

    def get(self, *, userId: str, id: str, format: str):
        assert userId == "me"
        assert format == "raw"
        return FakeMessageGetRequest(self.payloads[id])


class FakeUsersResource:
    def __init__(self, email: str, payloads: dict[str, dict[str, object]]) -> None:
        self.email = email
        self.messages_resource = FakeMessagesResource(payloads)

    def getProfile(self, userId: str):
        assert userId == "me"
        return FakeProfileRequest(self.email)

    def messages(self) -> FakeMessagesResource:
        return self.messages_resource


class FakeService:
    def __init__(self, email: str, payloads: dict[str, dict[str, object]]) -> None:
        self.users_resource = FakeUsersResource(email, payloads)

    def users(self) -> FakeUsersResource:
        return self.users_resource


def fake_google_modules(*, service: FakeService | None = None) -> dict[str, object]:
    return {
        "Request": SimpleNamespace,
        "Credentials": FakeCredentials,
        "HttpError": RuntimeError,
        "InstalledAppFlow": FakeInstalledAppFlow,
        "build": (lambda *_args, **_kwargs: service),
    }


class GmailOAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="spamguard_gmail_oauth_"))
        self.credentials_path = self.temp_dir / "credentials.json"
        self.token_path = self.temp_dir / "gmail_token.json"
        self.destination_dir = self.temp_dir / "gmail"
        self.credentials_path.write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        FakeCredentials.from_authorized_user_file_result = None
        FakeInstalledAppFlow.next_flow = None

    def test_missing_credentials_file_raises_clear_error(self) -> None:
        with patch("spamguard_demo.gmail._google_modules", return_value=fake_google_modules()):
            with self.assertRaises(FileNotFoundError):
                get_gmail_credentials(credentials_path=self.temp_dir / "missing.json", token_path=self.token_path)

    def test_save_gmail_credentials_file_validates_and_writes_json(self) -> None:
        raw_bytes = json.dumps(
            {
                "installed": {
                    "client_id": "client-id",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        ).encode("utf-8")

        written_path = save_gmail_credentials_file(raw_bytes, self.credentials_path)

        self.assertEqual(written_path, self.credentials_path)
        payload = json.loads(self.credentials_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["installed"]["client_id"], "client-id")

    def test_save_gmail_credentials_file_rejects_invalid_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid credentials.json"):
            save_gmail_credentials_file(b'{"not_installed": true}', self.credentials_path)

    def test_valid_token_skips_browser_flow(self) -> None:
        valid_creds = FakeCredentials(valid=True, expired=False, refresh_token="refresh")
        FakeCredentials.from_authorized_user_file_result = valid_creds
        FakeInstalledAppFlow.next_flow = FakeFlow(FakeCredentials(valid=True, expired=False, refresh_token="refresh"))
        self.token_path.write_text("{}", encoding="utf-8")

        with patch("spamguard_demo.gmail._google_modules", return_value=fake_google_modules()):
            creds = get_gmail_credentials(
                credentials_path=self.credentials_path,
                token_path=self.token_path,
            )

        self.assertIs(creds, valid_creds)
        self.assertEqual(FakeInstalledAppFlow.next_flow.run_count, 0)

    def test_expired_token_refreshes_without_browser(self) -> None:
        expired_creds = FakeCredentials(valid=False, expired=True, refresh_token="refresh")
        FakeCredentials.from_authorized_user_file_result = expired_creds
        FakeInstalledAppFlow.next_flow = FakeFlow(FakeCredentials(valid=True, expired=False, refresh_token="refresh"))
        self.token_path.write_text("{}", encoding="utf-8")

        with patch("spamguard_demo.gmail._google_modules", return_value=fake_google_modules()):
            creds = get_gmail_credentials(
                credentials_path=self.credentials_path,
                token_path=self.token_path,
            )

        self.assertIs(creds, expired_creds)
        self.assertTrue(expired_creds.refreshed)
        self.assertEqual(FakeInstalledAppFlow.next_flow.run_count, 0)

    def test_refresh_failure_falls_back_to_browser_login(self) -> None:
        expired_creds = FakeCredentials(valid=False, expired=True, refresh_token="refresh")
        expired_creds.refresh_raises = True
        browser_creds = FakeCredentials(valid=True, expired=False, refresh_token="refresh", email="demo@example.com")
        FakeCredentials.from_authorized_user_file_result = expired_creds
        FakeInstalledAppFlow.next_flow = FakeFlow(browser_creds)
        self.token_path.write_text("{}", encoding="utf-8")

        with patch("spamguard_demo.gmail._google_modules", return_value=fake_google_modules()):
            creds = get_gmail_credentials(
                credentials_path=self.credentials_path,
                token_path=self.token_path,
            )

        self.assertIs(creds, browser_creds)
        self.assertEqual(FakeInstalledAppFlow.next_flow.run_count, 1)
        self.assertTrue(self.token_path.exists())

    def test_import_from_gmail_saves_snapshots_and_deduplicates_by_remote_id(self) -> None:
        raw_bytes = (
            b"From: alpha@example.com\r\n"
            b"To: you@example.com\r\n"
            b"Subject: Offer\r\n"
            b"Message-ID: <alpha@example.com>\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
            b"Offer body"
        )
        raw_value = base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")
        payloads = {
            "msg-1": {
                "id": "msg-1",
                "threadId": "thread-1",
                "raw": raw_value,
                "payload": {"headers": [{"name": "Subject", "value": "Offer"}]},
            },
            "msg-2": {
                "id": "msg-2",
                "threadId": "thread-2",
                "raw": raw_value,
                "payload": {"headers": [{"name": "Subject", "value": "Offer Duplicate"}]},
            },
        }
        service = FakeService("demo@example.com", payloads)
        valid_creds = FakeCredentials(valid=True, expired=False, refresh_token="refresh")
        FakeCredentials.from_authorized_user_file_result = valid_creds
        self.token_path.write_text("{}", encoding="utf-8")

        with patch("spamguard_demo.gmail._google_modules", return_value=fake_google_modules(service=service)):
            imported = import_from_gmail(
                max_messages=10,
                destination_dir=self.destination_dir,
                credentials_path=self.credentials_path,
                token_path=self.token_path,
            )

        self.assertEqual(len(imported), 1)
        self.assertTrue(imported[0].exists())
        source_index = load_source_index(self.destination_dir)
        self.assertEqual(source_index["provider"], "gmail")
        self.assertEqual(source_index["account_email"], "demo@example.com")
        self.assertEqual(len(source_index["messages"]), 1)
        message_metadata = next(iter(source_index["messages"].values()))
        self.assertEqual(message_metadata["remote_message_id"], "msg-1")
        self.assertEqual(message_metadata["thread_id"], "thread-1")

    def test_disconnect_gmail_only_removes_token_file(self) -> None:
        self.token_path.write_text("{}", encoding="utf-8")
        disconnect_gmail(self.token_path)
        self.assertFalse(self.token_path.exists())
        self.assertTrue(self.credentials_path.exists())

    def test_remove_gmail_credentials_file_removes_only_credentials(self) -> None:
        self.credentials_path.write_text("{}", encoding="utf-8")
        remove_gmail_credentials_file(self.credentials_path)
        self.assertFalse(self.credentials_path.exists())


if __name__ == "__main__":
    unittest.main()
