# LAN Popup Chat (CLI)

Dead-simple local-only chat for popup cowork nights. Host starts a room; everyone on the same Wi‑Fi/LAN joins from their terminal. No internet services, no accounts.

![LAN popup pudding](./assets/pudding.png)

## Quick start (participant)
Download (no clone needed):
```bash
curl -L -o popup.py https://raw.githubusercontent.com/weisserj/flan-party/weisserj/lan-cli-chat/popup.py
python popup.py join --host 192.168.50.23 --port 5678 --passphrase secret123
```

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
