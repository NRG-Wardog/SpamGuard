from __future__ import annotations

import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from spamguard_demo import (
    classify_mail_sources,
    collect_mail_items,
    connect_gmail,
    disconnect_gmail,
    ensure_demo_dirs,
    get_gmail_connection_status,
    import_from_gmail,
    remove_gmail_credentials_file,
    save_gmail_credentials_file,
    seed_demo_mailbox,
)

LANGS = {"he": "\u05e2\u05d1\u05e8\u05d9\u05ea", "en": "English"}
RTL_LANGS = {"he"}
TEXT = {
    "he": {
        "title": "SpamGuard Demo",
        "hero": "\u05d7\u05d9\u05d1\u05d5\u05e8 Gmail \u05d0\u05de\u05d9\u05ea\u05d9 + \u05e1\u05d9\u05d5\u05d5\u05d2 \u05de\u05d9\u05d9\u05dc\u05d9\u05dd",
        "sub": "\u05d4\u05d0\u05e4\u05dc\u05d9\u05e7\u05e6\u05d9\u05d4 \u05de\u05ea\u05d7\u05d1\u05e8\u05ea \u05dc-Google \u05d1-OAuth, \u05de\u05d5\u05e9\u05db\u05ea \u05d4\u05d5\u05d3\u05e2\u05d5\u05ea \u05de-INBOX, \u05e9\u05d5\u05de\u05e8\u05ea snapshot \u05de\u05e7\u05d5\u05de\u05d9, \u05d5\u05de\u05e1\u05d5\u05d5\u05d2\u05ea \u05d0\u05d5\u05ea\u05df.",
        "lang": "\u05e9\u05e4\u05d4",
        "lang_help": "\u05e2\u05d1\u05e8\u05d9\u05ea = RTL, English = LTR.",
        "connect": "Connect Gmail",
        "refresh": "Refresh Inbox",
        "disconnect": "Disconnect Gmail",
        "scan": "\u05e1\u05e4\u05d9\u05e8\u05ea \u05e7\u05d1\u05e6\u05d9\u05dd",
        "classify": "\u05d4\u05e8\u05e6\u05ea \u05e1\u05d9\u05d5\u05d5\u05d2",
        "controls": "\u05d4\u05d2\u05d3\u05e8\u05d5\u05ea",
        "oauth_upload": "\u05d4\u05e2\u05dc\u05d0\u05ea credentials.json \u05e9\u05dc Google OAuth",
        "save_oauth": "\u05e9\u05de\u05d9\u05e8\u05ea credentials.json",
        "remove_oauth": "\u05de\u05d7\u05d9\u05e7\u05ea credentials.json",
        "gmail_limit": "\u05db\u05de\u05d5\u05ea \u05de\u05d9\u05d9\u05dc\u05d9\u05dd \u05dc\u05e8\u05e2\u05e0\u05d5\u05df",
        "upload": "\u05d4\u05e2\u05dc\u05d0\u05ea \u05e7\u05d1\u05e6\u05d9 .eml \u05de\u05e7\u05d5\u05de\u05d9\u05d9\u05dd",
        "save_uploads": "\u05e9\u05de\u05d9\u05e8\u05ea \u05d4\u05e2\u05dc\u05d0\u05d5\u05ea",
        "seeded": "\u05e0\u05d5\u05e1\u05e4\u05d5 {count} \u05e7\u05d1\u05e6\u05d9 \u05d3\u05d5\u05d2\u05de\u05d4 \u05dc\u05ea\u05d9\u05d1\u05d4 \u05d4\u05de\u05e7\u05d5\u05de\u05d9\u05ea.",
        "uploads_saved": "\u05e0\u05e9\u05de\u05e8\u05d5 {count} \u05e7\u05d1\u05e6\u05d9 .eml \u05dc\u05ea\u05d9\u05d1\u05d4 \u05d4\u05de\u05e7\u05d5\u05de\u05d9\u05ea.",
        "connected": "\u05d4\u05d7\u05d9\u05d1\u05d5\u05e8 \u05dc-Gmail \u05d4\u05d5\u05e9\u05dc\u05dd \u05e2\u05d1\u05d5\u05e8 {email}.",
        "oauth_saved": "\u05e7\u05d5\u05d1\u05e5 credentials.json \u05e0\u05e9\u05de\u05e8 \u05d3\u05e8\u05da ה-UI.",
        "oauth_removed": "\u05e7\u05d5\u05d1\u05e5 credentials.json \u05e0\u05de\u05d7\u05e7 \u05de\u05d4\u05e4\u05e8\u05d5\u05d9\u05e7\u05d8.",
        "refreshed": "\u05d9\u05d5\u05d1\u05d0\u05d5 {count} \u05d4\u05d5\u05d3\u05e2\u05d5\u05ea \u05d7\u05d3\u05e9\u05d5\u05ea \u05de-Gmail.",
        "disconnected": "\u05d4-token \u05d4\u05de\u05e7\u05d5\u05de\u05d9 \u05e0\u05de\u05d7\u05e7. \u05d4\u05d7\u05e9\u05d1\u05d5\u05df \u05d1-Google \u05dc\u05d0 \u05e9\u05d5\u05e0\u05d4.",
        "credentials_missing_help": "\u05d4\u05e2\u05dc\u05d4 credentials.json \u05d3\u05e8\u05da \u05d4-UI \u05d0\u05d5 \u05d4\u05d5\u05e1\u05e3 \u05d0\u05d5\u05ea\u05d5 \u05dc\u05e9\u05d5\u05e8\u05e9 \u05d4\u05e8\u05d9\u05e4\u05d5 \u05db\u05d3\u05d9 \u05dc\u05d0\u05e4\u05e9\u05e8 Gmail OAuth.",
        "oauth_help": "\u05d4\u05e2\u05dc\u05d4 \u05db\u05d0\u05df \u05d0\u05ea \u05e7\u05d5\u05d1\u05e5 ה-Google OAuth Desktop App \u05d1\u05e9\u05dd credentials.json.",
        "scanned": "\u05e0\u05de\u05e6\u05d0\u05d5 {local_count} \u05e7\u05d1\u05e6\u05d9\u05dd \u05de\u05e7\u05d5\u05de\u05d9\u05d9\u05dd \u05d5-{gmail_count} \u05e7\u05d1\u05e6\u05d9 Gmail.",
        "classified": "\u05d4\u05e1\u05d9\u05d5\u05d5\u05d2 \u05d4\u05d5\u05e9\u05dc\u05dd. \u05e2\u05d5\u05d1\u05d3\u05d5 {count} \u05d4\u05d5\u05d3\u05e2\u05d5\u05ea.",
        "credentials": "credentials.json",
        "token": "OAuth token",
        "account": "\u05d7\u05e9\u05d1\u05d5\u05df",
        "session": "\u05de\u05e6\u05d1 \u05d7\u05d9\u05d1\u05d5\u05e8",
        "sync": "\u05e1\u05d9\u05e0\u05db\u05e8\u05d5\u05df \u05d0\u05d7\u05e8\u05d5\u05df",
        "snapshot_count": "\u05db\u05de\u05d5\u05ea snapshot",
        "yes": "\u05db\u05df",
        "no": "\u05dc\u05d0",
        "connected_state": "\u05de\u05d7\u05d5\u05d1\u05e8",
        "disconnected_state": "\u05dc\u05d0 \u05de\u05d7\u05d5\u05d1\u05e8",
        "total": "\u05e1\u05d4\"\u05db \u05de\u05d9\u05d9\u05dc\u05d9\u05dd",
        "spam": "Spam",
        "ham": "Ham",
        "pending": "\u05dc\u05dc\u05d0 \u05e1\u05d9\u05d5\u05d5\u05d2",
        "source_filter": "\u05de\u05e7\u05d5\u05e8",
        "status_filter": "\u05e1\u05d8\u05d8\u05d5\u05e1",
        "search": "\u05d7\u05d9\u05e4\u05d5\u05e9",
        "picker_help": "\u05d1\u05d7\u05d9\u05e8\u05ea \u05d4\u05de\u05d9\u05d9\u05dc \u05dc\u05ea\u05e6\u05d5\u05d2\u05d4 \u05de\u05e7\u05d3\u05d9\u05de\u05d4 \u05e0\u05e2\u05e9\u05d9\u05ea \u05db\u05d0\u05df.",
        "picker": "\u05d1\u05d7\u05e8 \u05de\u05d9\u05d9\u05dc \u05dc\u05ea\u05e6\u05d5\u05d2\u05d4",
        "table": "\u05ea\u05e6\u05d5\u05d2\u05ea \u05ea\u05d9\u05d1\u05d4",
        "preview": "\u05ea\u05e6\u05d5\u05d2\u05d4 \u05de\u05e7\u05d3\u05d9\u05de\u05d4",
        "none": "\u05d0\u05d9\u05df \u05d4\u05d5\u05d3\u05e2\u05d5\u05ea \u05dc\u05d4\u05e6\u05d2\u05d4.",
        "no_subject": "\u05dc\u05dc\u05d0 \u05e0\u05d5\u05e9\u05d0",
        "source": "\u05de\u05e7\u05d5\u05e8",
        "status": "\u05e1\u05d8\u05d8\u05d5\u05e1",
        "subject": "\u05e0\u05d5\u05e9\u05d0",
        "from": "\u05de\u05d0\u05ea",
        "to": "\u05d0\u05dc",
        "processed": "\u05e2\u05d5\u05d1\u05d3 \u05d1-",
        "synced_at": "\u05e1\u05d5\u05e0\u05db\u05e8\u05df \u05d1-",
        "provider": "\u05e1\u05e4\u05e7",
        "spam_pct": "\u05d0\u05d7\u05d5\u05d6 \u05e1\u05e4\u05d0\u05dd",
        "message_id": "Message-ID",
        "remote_id": "Remote ID",
        "thread_id": "Thread ID",
        "result": "\u05e0\u05ea\u05d9\u05d1 \u05ea\u05d5\u05e6\u05d0\u05d4",
        "original": "\u05e7\u05d5\u05d1\u05e5 \u05de\u05e7\u05d5\u05e8",
        "body": "\u05ea\u05d5\u05db\u05df",
        "local": "\u05de\u05e7\u05d5\u05de\u05d9",
        "gmail": "Gmail",
        "unclassified": "\u05dc\u05dc\u05d0 \u05e1\u05d9\u05d5\u05d5\u05d2",
    },
    "en": {
        "title": "SpamGuard Demo",
        "hero": "Real Gmail OAuth + Mail Classification",
        "sub": "The app connects to Google with OAuth, imports INBOX messages, stores a local snapshot, and classifies them.",
        "lang": "Language",
        "lang_help": "Hebrew = RTL, English = LTR.",
        "connect": "Connect Gmail",
        "refresh": "Refresh Inbox",
        "disconnect": "Disconnect Gmail",
        "scan": "Scan Local Folders",
        "classify": "Run Classification",
        "controls": "Controls",
        "oauth_upload": "Upload Google OAuth credentials.json",
        "save_oauth": "Save credentials.json",
        "remove_oauth": "Remove credentials.json",
        "gmail_limit": "Messages to refresh",
        "upload": "Upload local .eml files",
        "save_uploads": "Save Uploads",
        "seeded": "Seeded the local mailbox with {count} sample files.",
        "uploads_saved": "Saved {count} .eml files into the local mailbox.",
        "connected": "Connected to Gmail account {email}.",
        "oauth_saved": "Saved credentials.json through the UI.",
        "oauth_removed": "Removed credentials.json from the project.",
        "refreshed": "Imported {count} new Gmail messages.",
        "disconnected": "Removed the local OAuth token. The Google account itself was not changed.",
        "credentials_missing_help": "Upload `credentials.json` through the UI or place it in the repository root to enable Gmail OAuth.",
        "oauth_help": "Upload the Google OAuth Desktop App file here as credentials.json.",
        "scanned": "Found {local_count} local files and {gmail_count} Gmail files.",
        "classified": "Classification complete. Processed {count} messages.",
        "credentials": "credentials.json",
        "token": "OAuth token",
        "account": "Account",
        "session": "Session",
        "sync": "Last Sync",
        "snapshot_count": "Snapshot Count",
        "yes": "Yes",
        "no": "No",
        "connected_state": "Connected",
        "disconnected_state": "Disconnected",
        "total": "Total Mail",
        "spam": "Spam",
        "ham": "Ham",
        "pending": "Pending",
        "source_filter": "Source",
        "status_filter": "Status",
        "search": "Search",
        "picker_help": "Choose the message shown in the preview here.",
        "picker": "Preview Message",
        "table": "Mailbox",
        "preview": "Preview",
        "none": "No messages are available.",
        "no_subject": "No subject",
        "source": "Source",
        "status": "Status",
        "subject": "Subject",
        "from": "From",
        "to": "To",
        "processed": "Processed",
        "synced_at": "Synced At",
        "provider": "Provider",
        "spam_pct": "Spam %",
        "message_id": "Message-ID",
        "remote_id": "Remote ID",
        "thread_id": "Thread ID",
        "result": "Result Path",
        "original": "Original Path",
        "body": "Body",
        "local": "Local",
        "gmail": "Gmail",
        "unclassified": "UNCLASSIFIED",
    },
}


def tr(lang: str, key: str, **kwargs: object) -> str:
    text = TEXT[lang][key]
    return text.format(**kwargs) if kwargs else text


def direction(lang: str) -> tuple[str, str]:
    return ("rtl", "right") if lang in RTL_LANGS else ("ltr", "left")


def inject_css(dir_name: str, align: str) -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ direction: {dir_name}; text-align: {align}; }}
        [data-testid="stMarkdownContainer"], .stAlert {{ text-align: {align}; }}
        .hero, .card {{ direction: {dir_name}; text-align: {align}; }}
        .hero {{ background: linear-gradient(135deg, #15325e, #2f6da8); color: #fff7eb; border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }}
        .card {{ background: rgba(255,255,255,0.86); border: 1px solid rgba(21,50,94,0.1); border-radius: 18px; padding: 1rem; }}
        .label {{ color: #667489; font-size: 0.82rem; text-transform: uppercase; margin-top: 0.8rem; }}
        .value {{ color: #1f2937; font-size: 1rem; word-break: break-word; }}
        .body {{ white-space: pre-wrap; background: #f6f2ea; border-radius: 14px; padding: 0.9rem; margin-top: 0.5rem; }}
        .pill {{ display: inline-block; padding: 0.22rem 0.6rem; border-radius: 999px; font-weight: 700; margin-bottom: 0.8rem; }}
        .spam {{ color: #fff7f7; background: #a71d31; }} .ham {{ color: #effff8; background: #16774b; }} .pending {{ color: #fffaf0; background: #8b5d12; }}
        .stButton > button {{ border-radius: 999px; min-height: 2.8rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_state() -> None:
    defaults = {
        "gmail_max_messages": 20,
        "spam_threshold": 0.5,
        "ui_language": "he",
        "selected_mail_path": "",
        "selected_mail_path_widget": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def status_text(lang: str, prediction: str) -> str:
    if prediction in {"SPAM", "HAM"}:
        return prediction
    return tr(lang, "unclassified")


def source_text(lang: str, source: str) -> str:
    return tr(lang, "gmail") if source == "gmail" else tr(lang, "local")


def probability_text(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def save_uploads(uploaded_files: list[object], destination_dir: Path) -> int:
    saved = 0
    for uploaded_file in uploaded_files:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded_file.name).strip("._") or "mail.eml"
        if not safe_name.lower().endswith(".eml"):
            safe_name += ".eml"
        destination = destination_dir / safe_name
        suffix = 1
        while destination.exists():
            destination = destination_dir / f"{destination.stem}_{suffix}{destination.suffix}"
            suffix += 1
        destination.write_bytes(uploaded_file.getvalue())
        saved += 1
    return saved


def filtered_items(items: list[dict[str, object]], allowed_sources: list[str], allowed_statuses: list[str], search: str) -> list[dict[str, object]]:
    needle = search.strip().lower()
    kept = []
    for item in items:
        if item["source"] not in allowed_sources or item["prediction"] not in allowed_statuses:
            continue
        haystack = " ".join(
            [
                str(item.get("subject", "")),
                str(item.get("from", "")),
                str(item.get("text_preview", "")),
                str(item.get("remote_message_id", "")),
            ]
        ).lower()
        if needle and needle not in haystack:
            continue
        kept.append(item)
    return kept


def html_text(value: object) -> str:
    return escape(str(value or "-")).replace("\n", "<br>")


def render_preview(item: dict[str, object], lang: str) -> None:
    prediction = str(item.get("prediction") or "UNCLASSIFIED")
    pill_class = "pill spam" if prediction == "SPAM" else "pill ham" if prediction == "HAM" else "pill pending"
    st.markdown(
        f"""
        <section class="card">
            <div class="{pill_class}">{escape(status_text(lang, prediction))}</div>
            <div class="label">{escape(tr(lang, "subject"))}</div><div class="value">{html_text(item.get("subject") or tr(lang, "no_subject"))}</div>
            <div class="label">{escape(tr(lang, "from"))}</div><div class="value">{html_text(item.get("from"))}</div>
            <div class="label">{escape(tr(lang, "to"))}</div><div class="value">{html_text(item.get("to"))}</div>
            <div class="label">{escape(tr(lang, "provider"))}</div><div class="value">{html_text(item.get("provider"))}</div>
            <div class="label">{escape(tr(lang, "spam_pct"))}</div><div class="value">{escape(probability_text(item.get("spam_probability")))}</div>
            <div class="label">{escape(tr(lang, "message_id"))}</div><div class="value">{html_text(item.get("message_id"))}</div>
            <div class="label">{escape(tr(lang, "remote_id"))}</div><div class="value">{html_text(item.get("remote_message_id"))}</div>
            <div class="label">{escape(tr(lang, "thread_id"))}</div><div class="value">{html_text(item.get("thread_id"))}</div>
            <div class="label">{escape(tr(lang, "synced_at"))}</div><div class="value">{html_text(item.get("synced_at"))}</div>
            <div class="label">{escape(tr(lang, "result"))}</div><div class="value">{html_text(item.get("stored_result_path"))}</div>
            <div class="label">{escape(tr(lang, "original"))}</div><div class="value">{html_text(item.get("original_path"))}</div>
            <div class="label">{escape(tr(lang, "body"))}</div><div class="body">{html_text(item.get("body") or item.get("text_preview"))}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _sync_selected_mail_state(available_paths: list[str]) -> str:
    current_value = str(st.session_state.get("selected_mail_path", "") or "")
    widget_value = str(st.session_state.get("selected_mail_path_widget", "") or "")

    candidate = current_value or widget_value
    if candidate not in available_paths:
        candidate = available_paths[0] if available_paths else ""

    st.session_state["selected_mail_path"] = candidate
    st.session_state["selected_mail_path_widget"] = candidate
    return candidate


def main() -> None:
    st.set_page_config(page_title="SpamGuard Demo", layout="wide")
    default_state()

    dirs = ensure_demo_dirs()
    seeded = seed_demo_mailbox()

    lang_col1, lang_col2 = st.columns((5, 1.3))
    with lang_col2:
        st.selectbox("Language / \u05e9\u05e4\u05d4", options=list(LANGS), format_func=lambda code: LANGS[code], key="ui_language")
    lang = st.session_state["ui_language"]
    dir_name, align = direction(lang)
    inject_css(dir_name, align)

    with lang_col2:
        st.caption(tr(lang, "lang_help"))
    with lang_col1:
        st.markdown(
            f'<section class="hero"><div style="font-size:2rem;font-weight:800;">{escape(tr(lang, "hero"))}</div><div style="margin-top:0.7rem;">{escape(tr(lang, "sub"))}</div></section>',
            unsafe_allow_html=True,
        )

    gmail_status = get_gmail_connection_status(destination_dir=dirs["gmail_mail"])
    action_message = ""
    action_level = "info"
    if seeded:
        action_message = tr(lang, "seeded", count=len(seeded))
        action_level = "success"

    action_cols = st.columns(4)
    with action_cols[0]:
        connect_requested = st.button(tr(lang, "connect"), use_container_width=True, disabled=not gmail_status["credentials_exists"])
    with action_cols[1]:
        refresh_requested = st.button(tr(lang, "refresh"), use_container_width=True, disabled=not gmail_status["credentials_exists"])
    with action_cols[2]:
        disconnect_requested = st.button(tr(lang, "disconnect"), use_container_width=True, disabled=not gmail_status["token_exists"])
    with action_cols[3]:
        classify_requested = st.button(tr(lang, "classify"), use_container_width=True)

    with st.expander(tr(lang, "controls"), expanded=False):
        st.slider("Spam Threshold", min_value=0.10, max_value=0.90, step=0.05, key="spam_threshold")
        st.slider(tr(lang, "gmail_limit"), min_value=5, max_value=50, step=5, key="gmail_max_messages")
        oauth_file = st.file_uploader(
            tr(lang, "oauth_upload"),
            type=["json"],
            accept_multiple_files=False,
            help=tr(lang, "oauth_help"),
        )
        oauth_cols = st.columns(2)
        with oauth_cols[0]:
            save_oauth_requested = st.button(
                tr(lang, "save_oauth"),
                disabled=oauth_file is None,
            )
        with oauth_cols[1]:
            remove_oauth_requested = st.button(
                tr(lang, "remove_oauth"),
                disabled=not gmail_status["credentials_exists"],
            )
        uploaded = st.file_uploader(tr(lang, "upload"), type=["eml"], accept_multiple_files=True)
        save_requested = st.button(tr(lang, "save_uploads"), disabled=not uploaded)
        if save_oauth_requested and oauth_file is not None:
            try:
                save_gmail_credentials_file(oauth_file.getvalue())
                gmail_status = get_gmail_connection_status(destination_dir=dirs["gmail_mail"])
                action_message = tr(lang, "oauth_saved")
                action_level = "success"
            except Exception as exc:
                action_message = str(exc)
                action_level = "error"
        if remove_oauth_requested:
            remove_gmail_credentials_file()
            disconnect_gmail()
            gmail_status = get_gmail_connection_status(destination_dir=dirs["gmail_mail"])
            action_message = tr(lang, "oauth_removed")
            action_level = "info"
        if save_requested and uploaded:
            saved = save_uploads(uploaded, dirs["local_mail"])
            action_message = tr(lang, "uploads_saved", count=saved)
            action_level = "success"

    if connect_requested:
        try:
            profile = connect_gmail()
            gmail_status = get_gmail_connection_status(destination_dir=dirs["gmail_mail"])
            action_message = tr(lang, "connected", email=profile["email_address"] or "unknown")
            action_level = "success"
        except Exception as exc:
            action_message = str(exc)
            action_level = "error"

    if refresh_requested:
        try:
            imported = import_from_gmail(
                max_messages=st.session_state["gmail_max_messages"],
                destination_dir=dirs["gmail_mail"],
            )
            gmail_status = get_gmail_connection_status(destination_dir=dirs["gmail_mail"])
            action_message = tr(lang, "refreshed", count=len(imported))
            action_level = "success"
        except Exception as exc:
            action_message = str(exc)
            action_level = "error"

    if disconnect_requested:
        disconnect_gmail()
        gmail_status = get_gmail_connection_status(destination_dir=dirs["gmail_mail"])
        action_message = tr(lang, "disconnected")
        action_level = "info"

    if classify_requested:
        try:
            rows = classify_mail_sources(
                {"local": dirs["local_mail"], "gmail": dirs["gmail_mail"]},
                output_dir=dirs["result_root"],
                copy_mode="copy",
                threshold=st.session_state["spam_threshold"],
            )
            action_message = tr(lang, "classified", count=len(rows))
            action_level = "success"
        except Exception as exc:
            action_message = str(exc)
            action_level = "error"

    scan_cols = st.columns(4)
    with scan_cols[0]:
        st.metric(tr(lang, "credentials"), tr(lang, "yes") if gmail_status["credentials_exists"] else tr(lang, "no"))
    with scan_cols[1]:
        state = tr(lang, "connected_state") if gmail_status["connected"] else tr(lang, "disconnected_state")
        st.metric(tr(lang, "session"), state)
    with scan_cols[2]:
        st.metric(tr(lang, "account"), gmail_status.get("account_email") or "-")
    with scan_cols[3]:
        st.metric(tr(lang, "snapshot_count"), str(gmail_status.get("last_import_count", 0)))

    st.caption(f'{tr(lang, "sync")}: {gmail_status.get("last_sync_at") or "-"} | {tr(lang, "token")}: {tr(lang, "yes") if gmail_status["token_exists"] else tr(lang, "no")}')
    if not gmail_status["credentials_exists"]:
        st.warning(tr(lang, "credentials_missing_help"))

    if action_message:
        if action_level == "success":
            st.success(action_message)
        elif action_level == "error":
            st.error(action_message)
        else:
            st.info(action_message)

    items = collect_mail_items()
    total = len(items)
    spam_total = sum(1 for item in items if item["prediction"] == "SPAM")
    ham_total = sum(1 for item in items if item["prediction"] == "HAM")
    pending_total = sum(1 for item in items if item["prediction"] == "UNCLASSIFIED")
    metric_cols = st.columns(4)
    metric_cols[0].metric(tr(lang, "total"), str(total))
    metric_cols[1].metric(tr(lang, "spam"), str(spam_total))
    metric_cols[2].metric(tr(lang, "ham"), str(ham_total))
    metric_cols[3].metric(tr(lang, "pending"), str(pending_total))

    filter_cols = st.columns((1.1, 1.2, 1.7))
    with filter_cols[0]:
        allowed_sources = st.multiselect(tr(lang, "source_filter"), options=["local", "gmail"], default=["local", "gmail"], format_func=lambda value: source_text(lang, value))
    with filter_cols[1]:
        allowed_statuses = st.multiselect(tr(lang, "status_filter"), options=["SPAM", "HAM", "UNCLASSIFIED"], default=["SPAM", "HAM", "UNCLASSIFIED"], format_func=lambda value: status_text(lang, value))
    with filter_cols[2]:
        search_value = st.text_input(tr(lang, "search"))

    visible_items = filtered_items(items, allowed_sources, allowed_statuses, search_value)
    item_by_path = {str(item["original_path"]): item for item in visible_items}
    available_paths = list(item_by_path)
    selected_item = None
    if available_paths:
        selected_path = _sync_selected_mail_state(available_paths)
        st.caption(tr(lang, "picker_help"))
        st.selectbox(
            tr(lang, "picker"),
            options=available_paths,
            index=available_paths.index(selected_path),
            format_func=lambda path: f'[{source_text(lang, str(item_by_path[path]["source"]))}] {item_by_path[path]["subject"] or tr(lang, "no_subject")} | {status_text(lang, str(item_by_path[path]["prediction"]))}',
            key="selected_mail_path_widget",
        )
        current_widget_value = str(
            st.session_state.get("selected_mail_path_widget", selected_path) or selected_path
        )
        st.session_state["selected_mail_path"] = current_widget_value
        selected_item = item_by_path[current_widget_value]
    else:
        st.session_state["selected_mail_path"] = ""
        st.session_state["selected_mail_path_widget"] = ""

    table_col, preview_col = st.columns((1.35, 1))
    with table_col:
        st.subheader(tr(lang, "table"))
        if visible_items:
            rows = []
            for item in visible_items:
                rows.append(
                    {
                        tr(lang, "source"): source_text(lang, str(item["source"])),
                        tr(lang, "status"): status_text(lang, str(item["prediction"])),
                        tr(lang, "spam_pct"): probability_text(item["spam_probability"]),
                        tr(lang, "subject"): item["subject"] or tr(lang, "no_subject"),
                        tr(lang, "from"): item["from"] or "-",
                        tr(lang, "processed"): item["processed_at"] or "-",
                        tr(lang, "synced_at"): item["synced_at"] or "-",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(tr(lang, "none"))

    with preview_col:
        st.subheader(tr(lang, "preview"))
        if selected_item is not None:
            render_preview(selected_item, lang)
        else:
            st.info(tr(lang, "none"))


if __name__ == "__main__":
    main()
