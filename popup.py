#!/usr/bin/env python3
"""LAN-only popup chat for late-night coffeeshop events.

Run as host (admin):
  python popup.py host --port 5678 --session-name "Late Night" --summary-dir summaries

Run as participant:
  python popup.py join --host 192.168.1.23 --port 5678
"""
import argparse
import dataclasses
import datetime as dt
import json
import os
import queue
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LINE_END = "\n"
BUFFER_SIZE = 4096
PROFILE_CACHE = Path.home() / ".lanpopup_profile.json"


def utcnow() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def send_json(sock: socket.socket, payload: Dict) -> None:
    data = json.dumps(payload) + LINE_END
    sock.sendall(data.encode("utf-8"))


def recv_lines(sock: socket.socket, buffer: bytearray) -> List[Dict]:
    """Read from socket into buffer and return full JSON objects per line."""
    data = sock.recv(BUFFER_SIZE)
    if not data:
        raise ConnectionError("socket closed")
    buffer.extend(data)
    lines = buffer.split(b"\n")
    buffer.clear()
    if lines and lines[-1] != b"":
        buffer.extend(lines.pop())  # leftover partial line
    out = []
    for line in lines:
        if not line:
            continue
        out.append(json.loads(line.decode("utf-8")))
    return out


@dataclasses.dataclass
class Profile:
    name: str
    email: str
    twitter: Optional[str]
    github: Optional[str]
    working_on: str
    can_help_with: str
    want_to_talk_about: str

    @classmethod
    def from_input(cls) -> "Profile":
        prev = load_cached_profile()
        def prompt(label: str, default: Optional[str] = None, required: bool = True) -> str:
            msg = f"{label}" + (f" [{default}]" if default else "") + ": "
            while True:
                val = input(msg).strip()
                if not val and default:
                    val = default
                if required and not val:
                    print("This field is required.")
                    continue
                return val

        name = prompt("Name", prev.get("name") if prev else None)
        email = prompt("Email", prev.get("email") if prev else None)
        twitter = prompt("Twitter handle (optional)", prev.get("twitter") if prev else None, required=False)
        github = prompt("GitHub username (optional)", prev.get("github") if prev else None, required=False)
        working_on = prompt("What are you working on tonight?", prev.get("working_on") if prev else None)
        can_help_with = prompt("Who/how can you help?", prev.get("can_help_with") if prev else None)
        want_to_talk_about = prompt("What would you like to talk about?", prev.get("want_to_talk_about") if prev else None)
        prof = cls(name, email, twitter or None, github or None, working_on, can_help_with, want_to_talk_about)
        cache_profile(dataclasses.asdict(prof))
        return prof


def cache_profile(profile_dict: Dict) -> None:
    try:
        PROFILE_CACHE.write_text(json.dumps(profile_dict, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not cache profile ({exc})")


def load_cached_profile() -> Optional[Dict]:
    if PROFILE_CACHE.exists():
        try:
            return json.loads(PROFILE_CACHE.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


class Server:
    def __init__(self, host: str, port: int, session_name: str, summary_dir: Path, passphrase: Optional[str]):
        self.host = host
        self.port = port
        self.session_name = session_name
        self.summary_dir = summary_dir
        self.passphrase = passphrase
        self.server_sock: Optional[socket.socket] = None
        self.clients: Dict[socket.socket, Profile] = {}
        self.messages: List[Dict] = []
        self._shutdown = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(20)
        print(f"[host] Session '{self.session_name}' listening on {self.host}:{self.port}")
        threading.Thread(target=self._accept_loop, daemon=True).start()
        self._host_loop()

    def _accept_loop(self) -> None:
        assert self.server_sock
        while not self._shutdown.is_set():
            try:
                client_sock, addr = self.server_sock.accept()
            except OSError:
                break
            threading.Thread(target=self._client_thread, args=(client_sock, addr), daemon=True).start()

    def _client_thread(self, client_sock: socket.socket, addr: Tuple[str, int]) -> None:
        buffer = bytearray()
        try:
            # Expect hello
            messages = recv_lines(client_sock, buffer)
            if not messages or messages[0].get("type") != "hello":
                client_sock.close()
                return
            # Optional passphrase gate
            if self.passphrase:
                if messages[0].get("passphrase") != self.passphrase:
                    try:
                        send_json(client_sock, {"type": "system", "text": "wrong passphrase"})
                    except Exception:
                        pass
                    client_sock.close()
                    return

            profile = Profile(**messages[0]["profile"])
            with self._lock:
                self.clients[client_sock] = profile
            self._broadcast({"type": "system", "text": f"{profile.name} joined", "ts": utcnow()})
            print(f"[join] {profile.name} ({addr[0]})")
            while not self._shutdown.is_set():
                for msg in recv_lines(client_sock, buffer):
                    if msg.get("type") == "chat":
                        enriched = {
                            "type": "chat",
                            "ts": utcnow(),
                            "text": msg.get("text", ""),
                            "sender": dataclasses.asdict(profile),
                        }
                        with self._lock:
                            self.messages.append(enriched)
                        self._broadcast(enriched)
                    elif msg.get("type") == "leave":
                        raise ConnectionError("client requested leave")
        except Exception:
            pass
        finally:
            with self._lock:
                prof = self.clients.pop(client_sock, None)
            client_sock.close()
            if prof:
                self._broadcast({"type": "system", "text": f"{prof.name} left", "ts": utcnow()})
                print(f"[left] {prof.name} ({addr[0]})")

    def _broadcast(self, payload: Dict) -> None:
        dead = []
        for sock in list(self.clients.keys()):
            try:
                send_json(sock, payload)
            except Exception:
                dead.append(sock)
        for sock in dead:
            with self._lock:
                self.clients.pop(sock, None)
            try:
                sock.close()
            except Exception:
                pass

    def _host_loop(self) -> None:
        print("[host] Commands: /who, /save, /quit")
        while not self._shutdown.is_set():
            try:
                line = input()
            except EOFError:
                line = "/quit"
            if line.strip() == "/who":
                with self._lock:
                    names = [p.name for p in self.clients.values()]
                print(f"Active: {', '.join(names) if names else 'nobody yet'}")
            elif line.strip() == "/save":
                self.write_summary()
                print("Saved summary")
            elif line.strip() == "/quit":
                self._shutdown.set()
                break
        print("[host] Shutting down...")
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        self.write_summary()

    def write_summary(self) -> None:
        now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        summary_path = self.summary_dir / f"{now}_{self.session_name.replace(' ', '_')}_summary.md"
        email_path = summary_path.with_suffix(".email.txt")
        with self._lock:
            participants = list(self.clients.values())
            messages = list(self.messages)
        lines = [f"# {self.session_name} — transcript", ""]
        lines.append("## Participants")
        if participants:
            for p in participants:
                lines.append(f"- {p.name} <{p.email}> twitter: {p.twitter or '-'} github: {p.github or '-'}")
                lines.append(f"  • working on: {p.working_on}")
                lines.append(f"  • can help with: {p.can_help_with}")
                lines.append(f"  • wants to talk about: {p.want_to_talk_about}")
        else:
            lines.append("- none connected")
        lines.append("")
        lines.append("## Messages")
        if messages:
            for m in messages:
                sender = m.get("sender", {}).get("name", "?")
                lines.append(f"- [{m['ts']}] {sender}: {m['text']}")
        else:
            lines.append("- (no chat yet)")
        summary_path.write_text("\n".join(lines), encoding="utf-8")

        email_lines = [f"Subject: {self.session_name} recap", ""]
        email_lines.append("Thanks for joining the popup! Highlights and links below.\n")
        email_lines.append("People & projects:")
        for p in participants:
            email_lines.append(f"- {p.name}: {p.working_on} (can help with {p.can_help_with}; wants to talk about {p.want_to_talk_about})")
        email_lines.append("\nLinks & ideas from chat:")
        for m in messages:
            email_lines.append(f"- {m['text']}")
        email_lines.append("\nReply-all with updates or follow-ups!")
        email_path.write_text("\n".join(email_lines), encoding="utf-8")
        print(f"[host] Wrote {summary_path} and {email_path}")


class Client:
    def __init__(self, host: str, port: int, profile: Profile, passphrase: Optional[str]):
        self.host = host
        self.port = port
        self.profile = profile
        self.passphrase = passphrase
        self.sock: Optional[socket.socket] = None
        self._receiver_alive = threading.Event()

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        send_json(self.sock, {"type": "hello", "profile": dataclasses.asdict(self.profile), "passphrase": self.passphrase})
        print(f"Connected to {self.host}:{self.port}. Type your messages; /quit to leave.")
        self._receiver_alive.set()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._send_loop()

    def _recv_loop(self) -> None:
        buffer = bytearray()
        try:
            while self._receiver_alive.is_set():
                for msg in recv_lines(self.sock, buffer):
                    mtype = msg.get("type")
                    if mtype == "system":
                        print(f"[system] {msg.get('text')}")
                    elif mtype == "chat":
                        sender = msg.get("sender", {}).get("name", "?")
                        print(f"[{msg.get('ts')}] {sender}: {msg.get('text')}")
        except Exception:
            print("[disconnected]")
            self._receiver_alive.clear()

    def _send_loop(self) -> None:
        assert self.sock
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    line = "/quit\n"
                line = line.rstrip("\n")
                if line.strip() == "/quit":
                    send_json(self.sock, {"type": "leave"})
                    break
                if line.strip():
                    send_json(self.sock, {"type": "chat", "text": line})
        finally:
            self._receiver_alive.clear()
            try:
                self.sock.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAN popup chat")
    sub = parser.add_subparsers(dest="command", required=True)

    host_parser = sub.add_parser("host", help="start a popup session")
    host_parser.add_argument("--host", default="0.0.0.0", help="bind address (default all interfaces)")
    host_parser.add_argument("--port", type=int, default=5678)
    host_parser.add_argument("--session-name", default="Late Night Coffeeshop")
    host_parser.add_argument("--summary-dir", default="summaries")
    host_parser.add_argument("--passphrase", default=None, help="optional shared passphrase required to join")

    join_parser = sub.add_parser("join", help="join an existing popup session")
    join_parser.add_argument("--host", required=True, help="host IP on the LAN")
    join_parser.add_argument("--port", type=int, default=5678)
    join_parser.add_argument("--passphrase", default=None, help="passphrase if the host requires one")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "host":
        summary_dir = Path(args.summary_dir)
        server = Server(args.host, args.port, args.session_name, summary_dir, args.passphrase)
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        server.start()
    elif args.command == "join":
        profile = Profile.from_input()
        client = Client(args.host, args.port, profile, args.passphrase)
        client.start()


if __name__ == "__main__":
    main()
