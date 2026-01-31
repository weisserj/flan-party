# LAN Popup Chat (CLI)

Dead-simple local-only chat for popup cowork nights. Host starts a room; everyone on the same Wi‑Fi/LAN joins from their terminal. No internet services, no accounts.

## Quick start (host)
```bash
python popup.py host --host 192.168.50.23 --port 5678 --session-name "Late Night" --summary-dir summaries --passphrase secret123
```
Leave this window open. Commands: `/who`, `/save`, `/quit`.

## Quick start (participant)
```bash
python popup.py join --host 192.168.50.23 --port 5678 --passphrase secret123
```
First join asks for: name, email, optional Twitter/GitHub, what you’re working on, how you can help, what you want to talk about. Answers are cached locally.

## What you get
- LAN-only TCP chat; reachable only to devices on the same network.
- Host-controlled start/stop; nothing persists except generated summaries.
- Auto summaries (Markdown + email draft) on `/save` or `/quit` in `summaries/`.
- Optional shared passphrase to keep random LAN users out.

## Requirements
- Python 3.x
- Everyone must be on the same LAN/Wi‑Fi.

## Minimal safety tips
- Use a passphrase; share it only with attendees.
- Run on a trusted / event-only SSID. Quit the host when done.

---
Repo: flan-party/lan-popup-chat (public). Ready to use as-is.
