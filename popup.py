#!/usr/bin/env python3
"""LAN-only popup chat for late-night coffeeshop events.

Run as host (admin):
  python popup.py host --port 5678 --session-name "Late Night" --summary-dir summaries

Run as participant:
  python popup.py join --host 192.168.1.23 --port 5678

Features:
  - Session continuation with history replay on reconnect
  - Commands: /who, /whois, /help, /quit
  - Curses TUI with scrollable messages, status line, input bar
  - BLE proximity detection (with network RTT fallback)
"""
import argparse
import asyncio
import curses
import dataclasses
import datetime as dt
import json
import platform
import signal
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LINE_END = "\n"
BUFFER_SIZE = 4096
PROFILE_CACHE = Path.home() / ".lanpopup_profile.json"
SESSION_CACHE = Path.home() / ".lanpopup_session.json"


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


# --- Session cache helpers (Phase 1) ---

def load_session_cache() -> Optional[Dict]:
    if SESSION_CACHE.exists():
        try:
            return json.loads(SESSION_CACHE.read_text())
        except Exception:
            return None
    return None


def save_session_cache(host: str, port: int, session_id: str) -> None:
    try:
        SESSION_CACHE.write_text(json.dumps({
            "host": host, "port": port, "session_id": session_id
        }, indent=2))
    except Exception:
        pass


# --- Proximity Scanner (Phase 4) ---

class ProximityScanner:
    """Background BLE/RTT proximity scanner."""

    def __init__(self, send_report_cb, get_peer_ips_cb):
        self._send_report = send_report_cb
        self._get_peer_ips = get_peer_ips_cb
        self._stop = threading.Event()
        self._has_bleak = False
        try:
            import bleak  # noqa: F401
            self._has_bleak = True
        except ImportError:
            pass

    def start(self) -> None:
        t = threading.Thread(target=self._scan_loop, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def _scan_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._has_bleak:
                    self._ble_scan()
                else:
                    self._rtt_scan()
            except Exception:
                pass
            self._stop.wait(15)

    def _ble_scan(self) -> None:
        import bleak

        async def _do_scan():
            scanner = bleak.BleakScanner()
            devices = await scanner.discover(timeout=5.0)
            results = []
            for d in devices:
                results.append({
                    "addr": d.address,
                    "rssi": d.rssi if hasattr(d, "rssi") else None,
                    "name": d.name or "",
                })
            return results

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_do_scan())
        finally:
            loop.close()
        if results:
            self._send_report(results)

    def _rtt_scan(self) -> None:
        peer_ips = self._get_peer_ips()
        if not peer_ips:
            return
        results = []
        for ip in peer_ips:
            rtt = self._ping_rtt(ip)
            if rtt is not None:
                results.append({"addr": ip, "rssi": None, "name": "", "rtt_ms": rtt})
        if results:
            self._send_report(results)

    @staticmethod
    def _ping_rtt(ip: str) -> Optional[float]:
        try:
            flag = "-n" if platform.system().lower() == "windows" else "-c"
            result = subprocess.run(
                ["ping", flag, "1", "-W", "1", ip],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                if "time=" in line:
                    part = line.split("time=")[1]
                    ms = float(part.split()[0].replace("ms", ""))
                    return ms
        except Exception:
            pass
        return None


# --- Curses TUI (Phase 3) ---

class CursesUI:
    """Curses-based terminal UI with scrollable messages, status bar, and input line."""

    def __init__(self, on_input_cb):
        self._on_input = on_input_cb
        self._messages: List[str] = []
        self._msg_lock = threading.Lock()
        self._scroll_offset = 0
        self.status_text = ""
        self._input_buf = ""
        self._cursor_pos = 0
        self._running = True
        self._stdscr = None

    def add_message(self, text: str) -> None:
        with self._msg_lock:
            self._messages.append(text)
            self._scroll_offset = 0  # auto-scroll to bottom on new message

    def set_status(self, text: str) -> None:
        self.status_text = text

    def stop(self) -> None:
        self._running = False

    def run(self, stdscr) -> None:
        self._stdscr = stdscr
        curses.curs_set(1)
        curses.use_default_colors()
        stdscr.timeout(100)

        while self._running:
            try:
                self._draw(stdscr)
                ch = stdscr.getch()
                if ch == -1:
                    continue
                elif ch == curses.KEY_RESIZE:
                    continue
                elif ch in (curses.KEY_ENTER, 10, 13):
                    line = self._input_buf.strip()
                    self._input_buf = ""
                    self._cursor_pos = 0
                    if line:
                        self._on_input(line)
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    if self._cursor_pos > 0:
                        self._input_buf = (
                            self._input_buf[:self._cursor_pos - 1]
                            + self._input_buf[self._cursor_pos:]
                        )
                        self._cursor_pos -= 1
                elif ch == curses.KEY_LEFT:
                    if self._cursor_pos > 0:
                        self._cursor_pos -= 1
                elif ch == curses.KEY_RIGHT:
                    if self._cursor_pos < len(self._input_buf):
                        self._cursor_pos += 1
                elif ch == curses.KEY_UP:
                    with self._msg_lock:
                        max_scroll = max(0, len(self._messages) - 1)
                    self._scroll_offset = min(self._scroll_offset + 1, max_scroll)
                elif ch == curses.KEY_DOWN:
                    self._scroll_offset = max(0, self._scroll_offset - 1)
                elif ch == curses.KEY_DC:  # delete key
                    if self._cursor_pos < len(self._input_buf):
                        self._input_buf = (
                            self._input_buf[:self._cursor_pos]
                            + self._input_buf[self._cursor_pos + 1:]
                        )
                elif 32 <= ch <= 126:
                    self._input_buf = (
                        self._input_buf[:self._cursor_pos]
                        + chr(ch)
                        + self._input_buf[self._cursor_pos:]
                    )
                    self._cursor_pos += 1
            except curses.error:
                pass

    def _draw(self, stdscr) -> None:
        try:
            h, w = stdscr.getmaxyx()
            if h < 3 or w < 10:
                return

            msg_height = h - 2

            # Message area
            with self._msg_lock:
                all_lines = []
                for m in self._messages:
                    # Wrap long lines
                    while len(m) > w - 1:
                        all_lines.append(m[:w - 1])
                        m = m[w - 1:]
                    all_lines.append(m)

                total = len(all_lines)
                end = total - self._scroll_offset
                start = max(0, end - msg_height)
                visible = all_lines[start:end]

            for i in range(msg_height):
                try:
                    stdscr.move(i, 0)
                    stdscr.clrtoeol()
                    if i < len(visible):
                        stdscr.addnstr(i, 0, visible[i], w - 1)
                except curses.error:
                    pass

            # Status line (reverse video)
            status = self.status_text[:w - 1] if self.status_text else ""
            try:
                stdscr.move(h - 2, 0)
                stdscr.clrtoeol()
                stdscr.addnstr(h - 2, 0, status.ljust(w - 1), w - 1, curses.A_REVERSE)
            except curses.error:
                pass

            # Input line
            prompt_str = "> " + self._input_buf
            try:
                stdscr.move(h - 1, 0)
                stdscr.clrtoeol()
                stdscr.addnstr(h - 1, 0, prompt_str[:w - 1], w - 1)
                cursor_x = min(2 + self._cursor_pos, w - 1)
                stdscr.move(h - 1, cursor_x)
            except curses.error:
                pass

            stdscr.noutrefresh()
            curses.doupdate()
        except curses.error:
            pass


# --- Server ---

class Server:
    def __init__(self, host: str, port: int, session_name: str, summary_dir: Path, passphrase: Optional[str]):
        self.host = host
        self.port = port
        self.session_name = session_name
        self.summary_dir = summary_dir
        self.passphrase = passphrase
        self.server_sock: Optional[socket.socket] = None
        self.clients: Dict[socket.socket, Profile] = {}
        self.client_ips: Dict[socket.socket, str] = {}
        self.messages: List[Dict] = []
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        # Phase 1: session ID and participant tracking
        self.session_id = str(uuid.uuid4())
        self.all_participants: Dict[str, Profile] = {}  # "name:email" -> Profile
        # Phase 4: proximity
        self.client_ble_addr: Dict[str, str] = {}  # "name:email" -> ble addr
        self.client_scans: Dict[str, List[Dict]] = {}  # "name:email" -> scan results

    def start(self) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(20)
        print(f"[host] Session '{self.session_name}' listening on {self.host}:{self.port}")
        print(f"[host] Session ID: {self.session_id}")
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

    def _participant_key(self, profile: Profile) -> str:
        return f"{profile.name}:{profile.email}"

    def _build_participants_msg(self) -> Dict:
        with self._lock:
            profiles = []
            for sock, p in self.clients.items():
                d = dataclasses.asdict(p)
                d["ip"] = self.client_ips.get(sock, "")
                profiles.append(d)
        return {"type": "participants", "profiles": profiles}

    def _broadcast_participants(self) -> None:
        self._broadcast(self._build_participants_msg())

    def _compute_proximity(self, key: str) -> List[Dict]:
        """Cross-reference BLE scans to find who is near whom."""
        with self._lock:
            my_scans = self.client_scans.get(key, [])
            if not my_scans:
                return []
            # Build addr -> name mapping from all clients' BLE addresses
            addr_to_name: Dict[str, str] = {}
            for k, addr in self.client_ble_addr.items():
                if k != key:
                    name = k.split(":")[0]
                    addr_to_name[addr.upper()] = name
            # Match scanned addresses
            nearby = []
            for scan in my_scans:
                addr = scan.get("addr", "").upper()
                if addr in addr_to_name:
                    nearby.append({
                        "name": addr_to_name[addr],
                        "rssi": scan.get("rssi"),
                        "rtt_ms": scan.get("rtt_ms"),
                    })
            # Also handle RTT-based proximity (match by IP)
            ip_to_name: Dict[str, str] = {}
            for sock, prof in self.clients.items():
                k2 = self._participant_key(prof)
                if k2 != key:
                    ip = self.client_ips.get(sock, "")
                    if ip:
                        ip_to_name[ip] = prof.name
            for scan in my_scans:
                addr = scan.get("addr", "")
                if addr in ip_to_name and not any(n["name"] == ip_to_name[addr] for n in nearby):
                    nearby.append({
                        "name": ip_to_name[addr],
                        "rssi": scan.get("rssi"),
                        "rtt_ms": scan.get("rtt_ms"),
                    })
            # Sort by RSSI (higher=closer) or RTT (lower=closer)
            def sort_key(item):
                if item.get("rssi") is not None:
                    return -item["rssi"]  # higher RSSI = closer, so negate for ascending
                if item.get("rtt_ms") is not None:
                    return item["rtt_ms"]
                return 9999
            nearby.sort(key=sort_key)
            return nearby[:5]

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
            key = self._participant_key(profile)
            client_session_id = messages[0].get("session_id")

            # Detect reconnect
            is_reconnect = (
                client_session_id == self.session_id
                and key in self.all_participants
            )

            # Send history before welcome on reconnect
            if is_reconnect:
                with self._lock:
                    history = list(self.messages)
                send_json(client_sock, {"type": "history", "messages": history})

            # Send welcome
            send_json(client_sock, {
                "type": "welcome",
                "session_id": self.session_id,
                "session_name": self.session_name,
            })

            with self._lock:
                self.clients[client_sock] = profile
                self.client_ips[client_sock] = addr[0]
                self.all_participants[key] = profile

            if is_reconnect:
                self._broadcast({"type": "system", "text": f"{profile.name} reconnected", "ts": utcnow()})
                print(f"[reconnect] {profile.name} ({addr[0]})")
            else:
                self._broadcast({"type": "system", "text": f"{profile.name} joined", "ts": utcnow()})
                print(f"[join] {profile.name} ({addr[0]})")

            self._broadcast_participants()

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
                    elif msg.get("type") == "proximity_report":
                        scans = msg.get("scans", [])
                        ble_addr = msg.get("ble_addr", "")
                        with self._lock:
                            self.client_scans[key] = scans
                            if ble_addr:
                                self.client_ble_addr[key] = ble_addr
                        # Compute and send proximity updates to all clients
                        self._send_proximity_updates()
        except Exception:
            pass
        finally:
            with self._lock:
                prof = self.clients.pop(client_sock, None)
                self.client_ips.pop(client_sock, None)
            client_sock.close()
            if prof:
                self._broadcast({"type": "system", "text": f"{prof.name} left", "ts": utcnow()})
                print(f"[left] {prof.name} ({addr[0]})")
                self._broadcast_participants()

    def _send_proximity_updates(self) -> None:
        """Send proximity_update to each connected client."""
        with self._lock:
            clients_snapshot = list(self.clients.items())
        for sock, prof in clients_snapshot:
            key = self._participant_key(prof)
            nearby = self._compute_proximity(key)
            try:
                send_json(sock, {"type": "proximity_update", "nearby": nearby})
            except Exception:
                pass

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
                self.client_ips.pop(sock, None)
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


# --- Client ---

class Client:
    def __init__(self, host: str, port: int, profile: Profile, passphrase: Optional[str], use_tui: bool = True):
        self.host = host
        self.port = port
        self.profile = profile
        self.passphrase = passphrase
        self.use_tui = use_tui
        self.sock: Optional[socket.socket] = None
        self._receiver_alive = threading.Event()
        self._send_lock = threading.Lock()
        self._participants: List[Dict] = []
        self._participants_lock = threading.Lock()
        self.nearby: List[Dict] = []
        self._nearby_lock = threading.Lock()
        self.session_id: Optional[str] = None
        self.session_name: Optional[str] = None
        self._ui: Optional[CursesUI] = None

    def _send_json(self, payload: Dict) -> None:
        with self._send_lock:
            send_json(self.sock, payload)

    def _display(self, text: str) -> None:
        if self._ui:
            self._ui.add_message(text)
        else:
            print(text)

    def _update_status(self) -> None:
        if not self._ui:
            return
        parts = []
        if self.session_name:
            parts.append(self.session_name)
        with self._participants_lock:
            count = len(self._participants)
        parts.append(f"{count} online")
        with self._nearby_lock:
            if self.nearby:
                names = ", ".join(n["name"] for n in self.nearby)
                parts.append(f"nearby: {names}")
        self._ui.set_status(" | ".join(parts))

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

        # Check for cached session ID for reconnect
        cached = load_session_cache()
        cached_session_id = None
        if cached and cached.get("host") == self.host and cached.get("port") == self.port:
            cached_session_id = cached.get("session_id")

        hello = {
            "type": "hello",
            "profile": dataclasses.asdict(self.profile),
            "passphrase": self.passphrase,
        }
        if cached_session_id:
            hello["session_id"] = cached_session_id

        send_json(self.sock, hello)

        self._receiver_alive.set()
        threading.Thread(target=self._recv_loop, daemon=True).start()

        # Start proximity scanner
        scanner = ProximityScanner(
            send_report_cb=self._send_proximity_report,
            get_peer_ips_cb=self._get_peer_ips,
        )
        scanner.start()

        if self.use_tui:
            self._ui = CursesUI(on_input_cb=self._process_input)
            # Run curses on main thread
            try:
                curses.wrapper(self._ui.run)
            finally:
                scanner.stop()
                self._cleanup()
        else:
            print(f"Connected to {self.host}:{self.port}. Type your messages; /quit to leave.")
            try:
                self._send_loop()
            finally:
                scanner.stop()

    def _send_proximity_report(self, scans: List[Dict]) -> None:
        if not self._receiver_alive.is_set():
            return
        try:
            self._send_json({"type": "proximity_report", "scans": scans})
        except Exception:
            pass

    def _get_peer_ips(self) -> List[str]:
        with self._participants_lock:
            return [
                p.get("ip", "")
                for p in self._participants
                if p.get("ip") and p.get("name") != self.profile.name
            ]

    def _recv_loop(self) -> None:
        buffer = bytearray()
        try:
            while self._receiver_alive.is_set():
                for msg in recv_lines(self.sock, buffer):
                    mtype = msg.get("type")
                    if mtype == "system":
                        self._display(f"[system] {msg.get('text')}")
                    elif mtype == "chat":
                        sender = msg.get("sender", {}).get("name", "?")
                        ts = msg.get("ts", "")
                        # Show short time if available
                        short_ts = ts
                        if "T" in ts:
                            short_ts = ts.split("T")[1].replace("Z", "")[:5]
                        self._display(f"[{short_ts}] {sender}: {msg.get('text')}")
                    elif mtype == "welcome":
                        self.session_id = msg.get("session_id")
                        self.session_name = msg.get("session_name")
                        if self.session_id:
                            save_session_cache(self.host, self.port, self.session_id)
                        self._display(f"[system] Welcome to '{self.session_name}'")
                        self._update_status()
                    elif mtype == "history":
                        hist_msgs = msg.get("messages", [])
                        if hist_msgs:
                            self._display("[system] --- message history ---")
                            for hm in hist_msgs:
                                sender = hm.get("sender", {}).get("name", "?")
                                ts = hm.get("ts", "")
                                short_ts = ts
                                if "T" in ts:
                                    short_ts = ts.split("T")[1].replace("Z", "")[:5]
                                self._display(f"[{short_ts}] {sender}: {hm.get('text')}")
                            self._display("[system] --- end history ---")
                    elif mtype == "participants":
                        profiles = msg.get("profiles", [])
                        with self._participants_lock:
                            self._participants = profiles
                        self._update_status()
                    elif mtype == "proximity_update":
                        nearby = msg.get("nearby", [])
                        with self._nearby_lock:
                            self.nearby = nearby
                        self._update_status()
        except Exception:
            self._display("[disconnected]")
            self._receiver_alive.clear()
            if self._ui:
                self._ui.stop()

    def _process_input(self, line: str) -> None:
        """Handle user input: commands or chat messages."""
        if line == "/quit":
            try:
                self._send_json({"type": "leave"})
            except Exception:
                pass
            self._receiver_alive.clear()
            if self._ui:
                self._ui.stop()
            return
        elif line == "/who":
            with self._participants_lock:
                names = [p.get("name", "?") for p in self._participants]
            if names:
                self._display(f"[who] Online ({len(names)}): {', '.join(names)}")
            else:
                self._display("[who] No participants info yet.")
            return
        elif line.startswith("/whois "):
            target = line[7:].strip()
            with self._participants_lock:
                found = None
                for p in self._participants:
                    if p.get("name", "").lower() == target.lower():
                        found = p
                        break
            if found:
                self._display(f"[whois] {found.get('name', '?')}")
                self._display(f"  Email: {found.get('email', '-')}")
                self._display(f"  Twitter: {found.get('twitter') or '-'}")
                self._display(f"  GitHub: {found.get('github') or '-'}")
                self._display(f"  Working on: {found.get('working_on', '-')}")
                self._display(f"  Can help with: {found.get('can_help_with', '-')}")
                self._display(f"  Wants to talk about: {found.get('want_to_talk_about', '-')}")
            else:
                self._display(f"[whois] No participant named '{target}'.")
            return
        elif line == "/help":
            self._display("[help] Available commands:")
            self._display("  /who        — list connected participants")
            self._display("  /whois NAME — show full profile for NAME")
            self._display("  /help       — show this help")
            self._display("  /quit       — leave the session")
            return
        elif line.startswith("/"):
            self._display(f"[system] Unknown command: {line.split()[0]}. Type /help for help.")
            return

        # Regular chat message
        try:
            self._send_json({"type": "chat", "text": line})
        except Exception:
            self._display("[error] Failed to send message.")

    def _send_loop(self) -> None:
        """Plain-text send loop (--no-tui mode)."""
        assert self.sock
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    line = "/quit\n"
                line = line.rstrip("\n")
                if line.strip():
                    self._process_input(line.strip())
                    if not self._receiver_alive.is_set():
                        break
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        self._receiver_alive.clear()
        try:
            if self.sock:
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
    join_parser.add_argument("--no-tui", action="store_true", help="disable curses TUI, use plain text")

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
        use_tui = not getattr(args, "no_tui", False)
        client = Client(args.host, args.port, profile, args.passphrase, use_tui=use_tui)
        client.start()


if __name__ == "__main__":
    main()
