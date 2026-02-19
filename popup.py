#!/usr/bin/env python3
"""Flan Party — popup social chat for your LAN.

Run as host (admin):
  python popup.py host --port 5678 --session-name "Hack Night" --summary-dir summaries

Run as participant:
  python popup.py join --host 192.168.1.23 --port 5678
"""
import argparse
import dataclasses
import datetime as dt
import json
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LINE_END = "\n"
BUFFER_SIZE = 4096
PROFILE_CACHE = Path.home() / ".lanpopup_profile.json"
MAX_MESSAGE_LEN = 500
MAX_MESSAGE_HISTORY = 1000
RATE_LIMIT_WINDOW_SECONDS = 5.0
RATE_LIMIT_MAX_MESSAGES = 5
IDENTITY_FIELD_LIMITS = {
    "name": 80,
    "email": 120,
    "twitter": 80,
    "github": 80,
}
ANSWER_MAX_LEN = 200
MAX_QUESTIONS = 10
MAX_QUESTION_KEY_LEN = 40
MAX_QUESTION_PROMPT_LEN = 200
DEFAULT_QUESTIONS: List[Dict] = [
    {"key": "working_on", "prompt": "What are you working on tonight?", "required": True},
    {"key": "can_help_with", "prompt": "Who/how can you help?", "required": True},
    {"key": "want_to_talk_about", "prompt": "What would you like to talk about?", "required": True},
]
CONTROL_CHAR_TRANSLATION = {i: None for i in range(32)}
CONTROL_CHAR_TRANSLATION[127] = None


def set_terminal_title(title: str) -> None:
    """Set the terminal window/tab title using ANSI escape codes."""
    if os.environ.get("FLAN_NO_TITLE") or not sys.stdout.isatty():
        return
    clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', title)
    sys.stdout.write(f"\033]0;{clean}\007")
    sys.stdout.flush()


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


def sanitize_text(value: str, max_len: Optional[int]) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.translate(CONTROL_CHAR_TRANSLATION)
    text = text.strip()
    if max_len is not None and len(text) > max_len:
        text = text[:max_len]
    return text


def sanitize_optional(value: Optional[str], max_len: int) -> Optional[str]:
    cleaned = sanitize_text(value or "", max_len)
    return cleaned or None


def validate_questions(questions: List[Dict]) -> List[Dict]:
    """Validate and return a cleaned list of question definitions."""
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"too many questions (max {MAX_QUESTIONS})")
    seen_keys: set = set()
    validated = []
    for q in questions:
        if not isinstance(q, dict):
            raise ValueError("each question must be an object")
        key = q.get("key", "")
        prompt = q.get("prompt", "")
        required = q.get("required", True)
        if not isinstance(key, str) or not re.match(r"^[a-z][a-z0-9_]*$", key):
            raise ValueError(f"invalid question key: {key!r} (must match [a-z][a-z0-9_]*)")
        if len(key) > MAX_QUESTION_KEY_LEN:
            raise ValueError(f"question key too long: {key!r}")
        if key in seen_keys:
            raise ValueError(f"duplicate question key: {key!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"question {key!r} has empty prompt")
        if len(prompt) > MAX_QUESTION_PROMPT_LEN:
            raise ValueError(f"question {key!r} prompt too long")
        seen_keys.add(key)
        validated.append({"key": key, "prompt": prompt.strip(), "required": bool(required)})
    return validated


def _slugify_key(prompt: str) -> str:
    """Turn a question prompt into a snake_case key."""
    key = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return key[:MAX_QUESTION_KEY_LEN] or "question"


def _dedupe_keys(questions: List[Dict]) -> List[Dict]:
    """Append _2, _3, etc. to duplicate keys."""
    seen: Dict[str, int] = {}
    out = []
    for q in questions:
        base = q["key"]
        if base in seen:
            seen[base] += 1
            q = {**q, "key": f"{base}_{seen[base]}"[:MAX_QUESTION_KEY_LEN]}
        else:
            seen[base] = 1
        out.append(q)
    return out


def prompt_host_setup(
    cli_session_name: Optional[str],
) -> Tuple[str, List[Dict]]:
    """Interactively prompt the host for session name and questions."""
    default_name = "Flan Party"
    if cli_session_name:
        session_name = cli_session_name
    else:
        val = input(f"Session name [{default_name}]: ").strip()
        session_name = val or default_name

    print("\nQuestions for participants (press Enter on blank line when done):\n")
    print("Defaults:")
    for i, q in enumerate(DEFAULT_QUESTIONS, 1):
        print(f"  {i}. {q['prompt']}")

    use_defaults = input("\nUse defaults? [Y/n]: ").strip().lower()
    if use_defaults in ("", "y", "yes"):
        return session_name, list(DEFAULT_QUESTIONS)

    questions: List[Dict] = []
    num = 1
    while num <= MAX_QUESTIONS:
        prompt_text = input(f"Question {num}: ").strip()
        if not prompt_text:
            break
        if len(prompt_text) > MAX_QUESTION_PROMPT_LEN:
            print(f"Please keep under {MAX_QUESTION_PROMPT_LEN} characters.")
            continue
        req = input("Required? [Y/n]: ").strip().lower()
        required = req in ("", "y", "yes")
        questions.append({
            "key": _slugify_key(prompt_text),
            "prompt": prompt_text,
            "required": required,
        })
        num += 1

    if not questions:
        print("No questions defined, using defaults.")
        return session_name, list(DEFAULT_QUESTIONS)

    questions = _dedupe_keys(questions)
    print(f"({len(questions)} question{'s' if len(questions) != 1 else ''} defined)")
    return session_name, validate_questions(questions)


def load_questions_file(path: str) -> List[Dict]:
    """Load and validate questions from a JSON file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read questions file: {exc}") from exc
    return validate_questions(data)


@dataclasses.dataclass
class Profile:
    name: str
    email: str
    twitter: Optional[str]
    github: Optional[str]
    answers: Dict[str, str]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "email": self.email,
            "twitter": self.twitter,
            "github": self.github,
            "answers": dict(self.answers),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Profile":
        return cls(
            name=data.get("name", "unknown"),
            email=data.get("email", "unknown"),
            twitter=data.get("twitter"),
            github=data.get("github"),
            answers=data.get("answers", {}),
        )

    @classmethod
    def from_input(cls, questions: List[Dict]) -> "Profile":
        prev = load_cached_profile()
        prev_answers = (prev.get("answers", {}) if prev else {})

        def ask(
            label: str,
            default: Optional[str] = None,
            required: bool = True,
            max_len: Optional[int] = None,
        ) -> str:
            msg = f"{label}" + (f" [{default}]" if default else "") + ": "
            while True:
                val = input(msg).strip()
                if not val and default:
                    val = default
                if required and not val:
                    print("This field is required.")
                    continue
                if max_len is not None and len(val) > max_len:
                    print(f"Please keep this under {max_len} characters.")
                    continue
                return val

        name = ask("Name", prev.get("name") if prev else None, max_len=IDENTITY_FIELD_LIMITS["name"])
        # If a different person is using the same machine, don't suggest the old profile
        if prev and name != prev.get("name"):
            prev = None
            prev_answers = {}
        email = ask("Email", prev.get("email") if prev else None, max_len=IDENTITY_FIELD_LIMITS["email"])
        twitter = ask(
            "Twitter handle (optional)",
            prev.get("twitter") if prev else None,
            required=False,
            max_len=IDENTITY_FIELD_LIMITS["twitter"],
        )
        github = ask(
            "GitHub username (optional)",
            prev.get("github") if prev else None,
            required=False,
            max_len=IDENTITY_FIELD_LIMITS["github"],
        )

        answers: Dict[str, str] = {}
        for q in questions:
            val = ask(
                q["prompt"],
                prev_answers.get(q["key"]),
                required=q.get("required", True),
                max_len=ANSWER_MAX_LEN,
            )
            if val:
                answers[q["key"]] = val

        prof = cls(name, email, twitter or None, github or None, answers)
        cache_profile(prof.to_dict())
        return prof


def cache_profile(profile_dict: Dict) -> None:
    try:
        PROFILE_CACHE.write_text(json.dumps(profile_dict, indent=2))
        os.chmod(PROFILE_CACHE, 0o600)
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
    def __init__(self, host: str, port: int, session_name: str, summary_dir: Path,
                 passphrase: Optional[str], questions: List[Dict]):
        self.host = host
        self.port = port
        self.session_name = session_name
        self.summary_dir = summary_dir
        self.passphrase = passphrase
        self.questions = questions
        self.question_labels: Dict[str, str] = {q["key"]: q["prompt"] for q in questions}
        self.server_sock: Optional[socket.socket] = None
        self.clients: Dict[socket.socket, Profile] = {}
        self.departed: List[Profile] = []
        self.messages: List[Dict] = []
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._rate_limits: Dict[socket.socket, deque] = {}

    def start(self) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(20)
        print(f"[host] Session '{self.session_name}' listening on {self.host}:{self.port}")
        print(f"[host] Questions ({len(self.questions)}):")
        for i, q in enumerate(self.questions, 1):
            req = "required" if q.get("required") else "optional"
            print(f"  {i}. {q['prompt']} ({req})")
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
            # Send questions and session info to client
            send_json(client_sock, {"type": "welcome", "session_name": self.session_name, "questions": self.questions})
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

            hello = messages[0]
            identity = hello.get("identity", {})
            if not isinstance(identity, dict):
                identity = {}
            raw_answers = hello.get("answers", {})
            if not isinstance(raw_answers, dict):
                raw_answers = {}

            # Sanitize answers — only keep keys matching configured questions, in order
            answers: Dict[str, str] = {}
            for q in self.questions:
                val = sanitize_text(raw_answers.get(q["key"], ""), ANSWER_MAX_LEN)
                if val:
                    answers[q["key"]] = val

            profile = Profile(
                name=sanitize_text(identity.get("name", "unknown"), IDENTITY_FIELD_LIMITS["name"]) or "unknown",
                email=sanitize_text(identity.get("email", "unknown"), IDENTITY_FIELD_LIMITS["email"]) or "unknown",
                twitter=sanitize_optional(identity.get("twitter"), IDENTITY_FIELD_LIMITS["twitter"]),
                github=sanitize_optional(identity.get("github"), IDENTITY_FIELD_LIMITS["github"]),
                answers=answers,
            )
            with self._lock:
                self.clients[client_sock] = profile
                self._rate_limits[client_sock] = deque()
            self._broadcast({"type": "system", "text": f"{profile.name} joined", "ts": utcnow()})
            print(f"[join] {profile.name} ({addr[0]})")
            while not self._shutdown.is_set():
                for msg in recv_lines(client_sock, buffer):
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "chat":
                        if not self._allow_message(client_sock):
                            try:
                                send_json(client_sock, {"type": "system", "text": "rate limit: slow down"})
                            except Exception:
                                pass
                            continue
                        text = sanitize_text(msg.get("text", ""), MAX_MESSAGE_LEN)
                        if not text:
                            continue
                        enriched = {
                            "type": "chat",
                            "ts": utcnow(),
                            "text": text,
                            "sender": profile.to_dict(),
                        }
                        with self._lock:
                            self.messages.append(enriched)
                            if len(self.messages) > MAX_MESSAGE_HISTORY:
                                del self.messages[:-MAX_MESSAGE_HISTORY]
                        self._broadcast(enriched)
                    elif msg.get("type") == "leave":
                        raise ConnectionError("client requested leave")
        except Exception:
            pass
        finally:
            with self._lock:
                prof = self.clients.pop(client_sock, None)
                self._rate_limits.pop(client_sock, None)
                if prof:
                    self.departed.append(prof)
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

    def _allow_message(self, client_sock: socket.socket) -> bool:
        now = time.monotonic()
        with self._lock:
            history = self._rate_limits.get(client_sock)
            if history is None:
                history = deque()
                self._rate_limits[client_sock] = history
            while history and (now - history[0]) > RATE_LIMIT_WINDOW_SECONDS:
                history.popleft()
            if len(history) >= RATE_LIMIT_MAX_MESSAGES:
                return False
            history.append(now)
        return True

    def _host_loop(self) -> None:
        print("[host] Commands: /who, /save, /quit")
        while not self._shutdown.is_set():
            try:
                line = input()
            except EOFError:
                line = "/quit"
            if line.strip() == "/who":
                with self._lock:
                    active = [p.name for p in self.clients.values()]
                    left = [p.name for p in self.departed]
                print(f"Active: {', '.join(active) if active else 'nobody yet'}")
                if left:
                    print(f"Left: {', '.join(left)}")
            elif line.strip() == "/save":
                self.write_summary()
                print("Saved summary")
            elif line.strip() == "/quit":
                self._shutdown.set()
                break
        print("[host] Shutting down...")
        self._broadcast({"type": "shutdown", "text": f"{self.session_name} over. Hosted by Flan Party \u2014 start your own (and \u2b50\ufe0f star us on GitHub) https://github.com/weisserj/flan-party", "ts": utcnow()})
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
            participants = list(self.clients.values()) + list(self.departed)
            messages = list(self.messages)
        lines = [f"# {self.session_name} — transcript", ""]
        lines.append("## Participants")
        if participants:
            for p in participants:
                lines.append(f"- {p.name} <{p.email}> twitter: {p.twitter or '-'} github: {p.github or '-'}")
                for key, label in self.question_labels.items():
                    answer = p.answers.get(key, "-")
                    lines.append(f"  • {label}: {answer}")
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
        email_lines.append(f"Thanks for joining {self.session_name}!\n")
        email_lines.append("Who was there:")
        for p in participants:
            parts = "; ".join(f"{self.question_labels[k]}: {v}" for k, v in p.answers.items() if v)
            email_lines.append(f"- {p.name}: {parts}" if parts else f"- {p.name}")
        email_lines.append("\nFrom the chat:")
        for m in messages:
            email_lines.append(f"- {m['text']}")
        email_lines.append("\nSee you next time!")
        email_path.write_text("\n".join(email_lines), encoding="utf-8")
        print(f"[host] Wrote {summary_path} and {email_path}")


class Client:
    def __init__(self, host: str, port: int, passphrase: Optional[str]):
        self.host = host
        self.port = port
        self.passphrase = passphrase
        self.profile: Optional[Profile] = None
        self.sock: Optional[socket.socket] = None
        self._receiver_alive = threading.Event()
        self._unread_count = 0
        self._unread_senders: List[str] = []
        self._notif_lock = threading.Lock()

    def _add_unread(self, sender: str) -> None:
        with self._notif_lock:
            self._unread_count += 1
            if sender not in self._unread_senders and len(self._unread_senders) < 3:
                self._unread_senders.append(sender)
            self._update_title()

    def _clear_unread(self) -> None:
        with self._notif_lock:
            self._unread_count = 0
            self._unread_senders.clear()
            set_terminal_title('\u26f5 Flan Party')

    def _update_title(self) -> None:
        if self._unread_count > 0:
            people = ', '.join(self._unread_senders)
            set_terminal_title(f"{self._unread_count} \u2022 {people}")

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        # Receive question definitions from server
        buffer = bytearray()
        messages = recv_lines(self.sock, buffer)
        if not messages or messages[0].get("type") != "welcome":
            print("Error: unexpected server response")
            self.sock.close()
            return
        questions = messages[0].get("questions", DEFAULT_QUESTIONS)
        session_name = messages[0].get("session_name", "Flan Party")
        # Welcome greeting
        print(f"\nWelcome to {session_name} on Flan Party. This is a popup chat in CLI for local area networks.\n")
        self.profile = Profile.from_input(questions)
        hello = {
            "type": "hello",
            "identity": {
                "name": self.profile.name,
                "email": self.profile.email,
                "twitter": self.profile.twitter,
                "github": self.profile.github,
            },
            "answers": self.profile.answers,
            "passphrase": self.passphrase,
        }
        send_json(self.sock, hello)
        print("Type your messages; /quit to leave.")
        set_terminal_title('\u26f5 Flan Party')
        self._receiver_alive.set()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._send_loop()

    def _recv_loop(self) -> None:
        buffer = bytearray()
        clean_shutdown = False
        try:
            while self._receiver_alive.is_set():
                for msg in recv_lines(self.sock, buffer):
                    mtype = msg.get("type")
                    if mtype == "shutdown":
                        text = sanitize_text(msg.get("text", "Session ended"), MAX_MESSAGE_LEN)
                        print(f"\n{text}")
                        clean_shutdown = True
                        self._receiver_alive.clear()
                        return
                    elif mtype == "system":
                        text = sanitize_text(msg.get("text", ""), MAX_MESSAGE_LEN)
                        print(f"[system] {text}")
                    elif mtype == "chat":
                        sender = msg.get("sender", {}).get("name", "?")
                        text = sanitize_text(msg.get("text", ""), MAX_MESSAGE_LEN)
                        print(f"[{msg.get('ts')}] {sender}: {text}")
                        if sender != self.profile.name:
                            self._add_unread(sender)
        except Exception:
            if not clean_shutdown:
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
                if len(line) > MAX_MESSAGE_LEN:
                    print(f"[warning] Message too long; truncating to {MAX_MESSAGE_LEN} characters.")
                    line = line[:MAX_MESSAGE_LEN]
                if line.strip():
                    send_json(self.sock, {"type": "chat", "text": line})
                    self._clear_unread()
        finally:
            self._receiver_alive.clear()
            self._clear_unread()
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
    host_parser.add_argument("--session-name", default=None)
    host_parser.add_argument("--summary-dir", default="summaries")
    host_parser.add_argument("--passphrase", default=None, help="optional shared passphrase required to join")
    host_parser.add_argument("--questions", default=None, help="path to JSON file defining profile questions")

    join_parser = sub.add_parser("join", help="join an existing popup session")
    join_parser.add_argument("--host", required=True, help="host IP on the LAN")
    join_parser.add_argument("--port", type=int, default=5678)
    join_parser.add_argument("--passphrase", default=None, help="passphrase if the host requires one")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "host":
        if args.questions:
            try:
                questions = load_questions_file(args.questions)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            session_name = args.session_name or "Flan Party"
        else:
            session_name, questions = prompt_host_setup(args.session_name)
        summary_dir = Path(args.summary_dir)
        server = Server(args.host, args.port, session_name, summary_dir, args.passphrase, questions)
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        server.start()
    elif args.command == "join":
        client = Client(args.host, args.port, args.passphrase)
        client.start()


if __name__ == "__main__":
    main()
