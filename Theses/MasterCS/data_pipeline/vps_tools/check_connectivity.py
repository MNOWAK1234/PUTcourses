import argparse
import csv
import json
import logging
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import paramiko


DEFAULT_SERVERS_CSV = "servers.csv"

DEFAULT_ROOT_USER = "root"
DEFAULT_KEY_USER = "ubuntu"

TMUX_SESSION = "fide"
REMOTE_OUTPUT_DIR_NAME = "fide_standard_games_by_id"


# ============================================================
# Wycisz Paramiko.
# Bez tego przy chwilowych problemach SSH potrafi pluć:
# Exception (client): Error reading SSH protocol banner
# Traceback ...
# ============================================================

for logger_name in (
    "paramiko",
    "paramiko.transport",
    "paramiko.auth_handler",
    "paramiko.sftp",
):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.disabled = True
    logger.setLevel(logging.CRITICAL + 1)

try:
    paramiko.util.log_to_file(os.devnull, level=logging.CRITICAL)
except Exception:
    pass

_STDERR_LOCK = Lock()


@contextmanager
def suppress_stderr_during_paramiko_connect():
    """
    Paramiko czasem wypisuje traceback z osobnego wątku transportu.
    To nie zawsze przechodzi przez normalny logging, więc na czas connect()
    przekierowujemy stderr do nul. Lock powoduje, że kilka connectów naraz
    nie nadpisuje sobie sys.stderr.
    """
    with _STDERR_LOCK:
        old_stderr = sys.stderr
        devnull = open(os.devnull, "w", encoding="utf-8", errors="ignore")
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old_stderr
            devnull.close()


# ============================================================
# Helpers
# ============================================================

def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_first_pem() -> str:
    pem_files = sorted(script_dir().glob("*.pem"))
    if not pem_files:
        return ""
    return str(pem_files[0])


def short_error(e: Exception, limit: int = 180) -> str:
    text = f"{type(e).__name__}: {str(e)}"
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    if "Error reading SSH protocol banner" in text:
        text = "SSHException: Error reading SSH protocol banner"

    if "WinError 10060" in text:
        text = "TimeoutError: WinError 10060 connect timeout"

    if "WinError 10054" in text:
        text = "ConnectionResetError: WinError 10054 connection reset"

    if len(text) > limit:
        return text[:limit - 3] + "..."

    return text


def remote_base_dir(user: str) -> str:
    if user == "root":
        return "/root/fide_scraper"
    return f"/home/{user}/fide_scraper"


def remote_output_dir(user: str) -> str:
    return f"{remote_base_dir(user)}/{REMOTE_OUTPUT_DIR_NAME}"


def sudo_prefix(user: str) -> str:
    return "" if user == "root" else "sudo "


def new_ssh_client() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return ssh


# ============================================================
# CSV
# ============================================================

def load_servers(path: str):
    """
    Format:

        host,password
        1.2.3.4,HASLO
        3.72.75.163,

    Jeśli password jest podane: próbuje root + password.
    Jeśli password jest puste albo hasło nie działa: próbuje ubuntu + pierwszy *.pem.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie ma pliku: {path}")

    servers = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV jest pusty albo nie ma nagłówka")

        if "host" not in reader.fieldnames or "password" not in reader.fieldnames:
            raise ValueError("CSV musi mieć kolumny: host,password")

        for i, row in enumerate(reader, start=1):
            host = (row.get("host") or "").strip()
            password = (row.get("password") or "").strip()

            if not host:
                continue

            servers.append({
                "index": i,
                "host": host,
                "password": password,
            })

    return servers


# ============================================================
# SSH
# ============================================================

def connect_once_password(host: str, password: str, args):
    ssh = new_ssh_client()
    try:
        with suppress_stderr_during_paramiko_connect():
            ssh.connect(
                hostname=host,
                username=DEFAULT_ROOT_USER,
                password=password,
                timeout=args.connect_timeout,
                banner_timeout=args.banner_timeout,
                auth_timeout=args.auth_timeout,
                look_for_keys=False,
                allow_agent=False,
            )
        return ssh, DEFAULT_ROOT_USER, "password"
    except Exception:
        try:
            ssh.close()
        except Exception:
            pass
        raise


def connect_once_key(host: str, key_path: str, args):
    ssh = new_ssh_client()
    try:
        with suppress_stderr_during_paramiko_connect():
            ssh.connect(
                hostname=host,
                username=DEFAULT_KEY_USER,
                key_filename=key_path,
                timeout=args.connect_timeout,
                banner_timeout=args.banner_timeout,
                auth_timeout=args.auth_timeout,
                look_for_keys=False,
                allow_agent=False,
            )
        return ssh, DEFAULT_KEY_USER, "pem"
    except Exception:
        try:
            ssh.close()
        except Exception:
            pass
        raise


def retry_connect(fn, retries: int, sleep_seconds: int):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e

            if attempt < retries:
                time.sleep(sleep_seconds)

    raise last_error


def connect_ssh(server: dict, key_path: str, args):
    host = server["host"]
    password = server.get("password") or ""
    errors = []

    if password:
        try:
            return retry_connect(
                lambda: connect_once_password(host, password, args),
                retries=args.connect_retries,
                sleep_seconds=args.retry_sleep,
            )
        except Exception as e:
            errors.append(f"password={short_error(e, 100)}")

    if key_path:
        try:
            return retry_connect(
                lambda: connect_once_key(host, key_path, args),
                retries=args.connect_retries,
                sleep_seconds=args.retry_sleep,
            )
        except Exception as e:
            errors.append(f"pem={short_error(e, 100)}")
    else:
        errors.append("pem=missing")

    raise RuntimeError("; ".join(errors) if errors else "auth failed")


def run_command(ssh: paramiko.SSHClient, command: str, timeout=None):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


# ============================================================
# Status / setup
# ============================================================

def parse_key_values(out: str):
    values = {}

    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def check_remote_status(ssh: paramiko.SSHClient, user: str, timeout: int):
    rdir = remote_base_dir(user)
    out_dir = remote_output_dir(user)

    command = f"""
RDIR={shlex.quote(rdir)}
OUTDIR={shlex.quote(out_dir)}

if command -v tmux >/dev/null 2>&1; then
  echo tmux_cmd=yes
  if tmux has-session -t {shlex.quote(TMUX_SESSION)} >/dev/null 2>&1; then
    echo tmux_fide=yes
  else
    echo tmux_fide=no
  fi
else
  echo tmux_cmd=no
  echo tmux_fide=no
fi

if ps -eo pid,ppid,stat,cmd | grep '[s]craperFaster.py' >/dev/null 2>&1; then
  echo scraper=yes
else
  echo scraper=no
fi

if [ -d "$OUTDIR" ]; then
  echo output_dir=yes
  echo csv=$(find "$OUTDIR" -maxdepth 1 -type f -name '*.csv' 2>/dev/null | wc -l)
else
  echo output_dir=no
  echo csv=0
fi

if [ -d "$RDIR" ]; then
  echo remote_dir=yes
else
  echo remote_dir=no
fi

if [ -x "$RDIR/venv/bin/python" ]; then
  echo venv=yes
  if "$RDIR/venv/bin/python" -c "import requests" >/dev/null 2>&1; then
    echo requests=venv
    echo python_cmd="$RDIR/venv/bin/python"
  else
    echo requests=missing
    echo python_cmd=none
  fi
elif command -v python3 >/dev/null 2>&1 && python3 -c "import requests" >/dev/null 2>&1; then
  echo venv=no
  echo requests=system
  echo python_cmd=python3
else
  echo venv=no
  echo requests=missing
  echo python_cmd=none
fi
""".strip()

    code, out, err = run_command(ssh, command, timeout=timeout)

    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or f"status command failed code={code}")

    values = parse_key_values(out)

    defaults = {
        "tmux_cmd": "unknown",
        "tmux_fide": "unknown",
        "scraper": "unknown",
        "output_dir": "unknown",
        "csv": "0",
        "remote_dir": "unknown",
        "venv": "unknown",
        "requests": "missing",
        "python_cmd": "none",
    }

    defaults.update(values)
    return defaults


def prepare_requests_env(ssh: paramiko.SSHClient, user: str):
    """
    Minimalny setup tylko pod test connectivity.
    Odpalamy go WYŁĄCZNIE jeśli NIE ma aktywnej sesji tmux fide,
    żeby nie mieszać w środowisku maszyny, która już scrapuje.
    """
    rdir = remote_base_dir(user)
    sudo = sudo_prefix(user)

    command = f"""
set -e
RDIR={shlex.quote(rdir)}
mkdir -p "$RDIR"

if ! command -v python3 >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  {sudo}apt-get update -y >/dev/null 2>&1
  {sudo}apt-get install -y python3 >/dev/null 2>&1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  {sudo}apt-get update -y >/dev/null 2>&1
  {sudo}apt-get install -y python3-venv >/dev/null 2>&1
fi

if [ ! -d "$RDIR/venv" ]; then
  cd "$RDIR"
  python3 -m venv venv >/dev/null 2>&1
fi

cd "$RDIR"
. venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install requests >/dev/null 2>&1
""".strip()

    code, out, err = run_command(ssh, command, timeout=None)

    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or f"prepare failed code={code}")


# ============================================================
# FIDE test
# ============================================================

def check_fide_requests(
    ssh: paramiko.SSHClient,
    user: str,
    python_cmd: str,
    player_id: str,
    tries: int,
    request_timeout: int,
):
    """
    Test podobny do realnego flow scrapera:
    1) GET https://ratings.fide.com/
    2) POST https://ratings.fide.com/a_calculations.phtml action=2, plr_id=...
    3) GET https://ratings.fide.com/a_indv_calculations.php dla znalezionego periodu
    """
    rdir = remote_base_dir(user)

    py_code = r'''
import re
import time
import requests
from urllib.parse import unquote

BASE = "https://ratings.fide.com/"
PLAYER_ID = __PLAYER_ID__
TRIES = __TRIES__
REQUEST_TIMEOUT = __REQUEST_TIMEOUT__

def one_try():
    result = {
        "home": False,
        "post": False,
        "calc": False,
        "home_status": "-",
        "post_status": "-",
        "calc_status": "-",
        "home_seconds": "-",
        "post_seconds": "-",
        "calc_seconds": "-",
        "period": "-",
        "error": "-",
    }

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
    })

    try:
        t = time.time()
        response = session.get(BASE, timeout=REQUEST_TIMEOUT)
        result["home_status"] = str(response.status_code)
        result["home_seconds"] = f"{time.time() - t:.2f}"
        result["home"] = (200 <= response.status_code < 500 and len(response.text or "") > 0)
    except Exception as e:
        result["error"] = "home:" + repr(e)[:160].replace(" ", "_")
        return result

    try:
        t = time.time()
        response = session.post(
            BASE + "a_calculations.phtml",
            data={"action": "2", "plr_id": PLAYER_ID},
            headers={
                "Referer": BASE + f"profile/{PLAYER_ID}/calculations",
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
            timeout=REQUEST_TIMEOUT,
        )

        html = response.text or ""
        result["post_status"] = str(response.status_code)
        result["post_seconds"] = f"{time.time() - t:.2f}"
        result["post"] = (200 <= response.status_code < 500 and len(html) > 0)

        period = None

        match = re.search(
            r'calculations\.phtml\?id_number=' + re.escape(PLAYER_ID) + r'&period=([^&"\']+)&rating=0',
            html,
        )

        if match:
            period = unquote(match.group(1))
        else:
            match = re.search(r'period=([^&"\']+)', html)
            if match:
                period = unquote(match.group(1))

        if not period:
            result["error"] = "post:no_period_link"
            return result

        result["period"] = period

    except Exception as e:
        result["error"] = "post:" + repr(e)[:160].replace(" ", "_")
        return result

    try:
        t = time.time()
        response = session.get(
            BASE + "a_indv_calculations.php",
            params={
                "id_number": PLAYER_ID,
                "rating_period": result["period"],
                "t": 0,
            },
            headers={
                "Referer": BASE + f"calculations.phtml?id_number={PLAYER_ID}&period={result['period']}&rating=0",
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "*/*",
            },
            timeout=REQUEST_TIMEOUT,
        )

        text = response.text or ""
        result["calc_status"] = str(response.status_code)
        result["calc_seconds"] = f"{time.time() - t:.2f}"
        result["calc"] = (200 <= response.status_code < 500 and len(text) > 0)

        if not result["calc"]:
            result["error"] = "calc:empty_or_bad_status"

    except Exception as e:
        result["error"] = "calc:" + repr(e)[:160].replace(" ", "_")
        return result

    return result

home_ok = 0
post_ok = 0
calc_ok = 0
last = None

for i in range(TRIES):
    last = one_try()

    if last["home"]:
        home_ok += 1
    if last["post"]:
        post_ok += 1
    if last["calc"]:
        calc_ok += 1

    if i + 1 < TRIES:
        time.sleep(5)

first_bad = "-"
if home_ok < TRIES:
    first_bad = "home"
elif post_ok < TRIES:
    first_bad = "post"
elif calc_ok < TRIES:
    first_bad = "calc"

print(
    "RESULT "
    f"home={home_ok}/{TRIES} "
    f"post={post_ok}/{TRIES} "
    f"calc={calc_ok}/{TRIES} "
    f"first_bad={first_bad} "
    f"last_home_status={last['home_status']} "
    f"last_post_status={last['post_status']} "
    f"last_calc_status={last['calc_status']} "
    f"last_home_s={last['home_seconds']} "
    f"last_post_s={last['post_seconds']} "
    f"last_calc_s={last['calc_seconds']} "
    f"period={last['period']} "
    f"error={last['error']}"
)
'''.strip()

    py_code = (
        py_code
        .replace("__PLAYER_ID__", json.dumps(str(player_id)))
        .replace("__TRIES__", str(tries))
        .replace("__REQUEST_TIMEOUT__", str(request_timeout))
    )

    command = f"""
cd {shlex.quote(rdir)}
{shlex.quote(python_cmd)} - <<'REMOTE_PY'
{py_code}
REMOTE_PY
""".strip()

    total_timeout = tries * (request_timeout * 3 + 25) + 60
    code, out, err = run_command(ssh, command, timeout=total_timeout)

    result = {
        "status": "FIDE ERROR",
        "home": "0/0",
        "post": "0/0",
        "calc": "0/0",
        "first_bad": "unknown",
        "details": (err.strip() or out.strip() or f"code={code}")[:240],
    }

    if code != 0:
        return result

    result_line = ""

    for line in out.splitlines():
        if line.startswith("RESULT "):
            result_line = line.strip()
            break

    if not result_line:
        result["details"] = "no RESULT line"
        return result

    parts = {}

    for part in result_line.split()[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parts[key] = value

    result["home"] = parts.get("home", "0/0")
    result["post"] = parts.get("post", "0/0")
    result["calc"] = parts.get("calc", "0/0")
    result["first_bad"] = parts.get("first_bad", "unknown")

    detail_keys = [
        "last_home_status",
        "last_post_status",
        "last_calc_status",
        "last_home_s",
        "last_post_s",
        "last_calc_s",
        "period",
        "error",
    ]

    result["details"] = " ".join(f"{key}={parts.get(key, '-')}" for key in detail_keys)

    expected = f"{tries}/{tries}"

    if result["home"] == expected and result["post"] == expected and result["calc"] == expected:
        result["status"] = "FIDE OK"
    else:
        result["status"] = "FIDE NOT OK"

    return result


# ============================================================
# Per server
# ============================================================

def format_status_prefix(host: str, user: str, auth: str, status: dict):
    setup = status.get("requests", "missing")

    return (
        f"[{host}] AUTH OK {user}/{auth} | "
        f"tmux={status.get('tmux_fide', '?')} "
        f"scraper={status.get('scraper', '?')} "
        f"csv={status.get('csv', '?')} "
        f"setup={setup}"
    )


def process_server(server: dict, key_path: str, args):
    host = server["host"]
    ssh = None

    try:
        ssh, user, auth = connect_ssh(server, key_path, args)

        status = check_remote_status(ssh, user, args.command_timeout)
        prefix = format_status_prefix(host, user, auth, status)

        tmux_active = status.get("tmux_fide") == "yes"
        python_cmd = status.get("python_cmd", "none")
        requests_ready = python_cmd != "none" and status.get("requests") in ("venv", "system")

        if tmux_active:
            # Nie robimy prepare, bo maszyna może aktualnie scrapować.
            if not requests_ready:
                return (
                    f"{prefix} | FIDE SKIPPED "
                    f"reason=active_tmux_no_requests_env"
                )

            fide = check_fide_requests(
                ssh=ssh,
                user=user,
                python_cmd=python_cmd,
                player_id=args.player_id,
                tries=args.tries,
                request_timeout=args.request_timeout,
            )

            return (
                f"{prefix} | {fide['status']} "
                f"home={fide['home']} post={fide['post']} calc={fide['calc']} "
                f"first_bad={fide['first_bad']} | {fide['details']}"
            )

        # Brak sesji: możemy przygotować środowisko, jeśli trzeba.
        if not requests_ready:
            if args.no_prepare:
                return (
                    f"{prefix} | FIDE SKIPPED "
                    f"reason=no_requests_env_and_no_prepare"
                )

            try:
                prepare_requests_env(ssh, user)
            except Exception as e:
                return (
                    f"{prefix} | PREPARE FAILED | {short_error(e, 220)}"
                )

            status = check_remote_status(ssh, user, args.command_timeout)
            prefix = format_status_prefix(host, user, auth, status)
            python_cmd = status.get("python_cmd", "none")
            requests_ready = python_cmd != "none" and status.get("requests") in ("venv", "system")

            if not requests_ready:
                return (
                    f"{prefix} | PREPARE FAILED | requests still missing"
                )

        fide = check_fide_requests(
            ssh=ssh,
            user=user,
            python_cmd=python_cmd,
            player_id=args.player_id,
            tries=args.tries,
            request_timeout=args.request_timeout,
        )

        return (
            f"{prefix} | {fide['status']} "
            f"home={fide['home']} post={fide['post']} calc={fide['calc']} "
            f"first_bad={fide['first_bad']} | {fide['details']}"
        )

    except Exception as e:
        return f"[{host}] AUTH/TEST FAILED | {short_error(e, 260)}"

    finally:
        if ssh is not None:
            try:
                ssh.close()
            except Exception:
                pass


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SSH/tmux/setup/FIDE checker. "
            "Jeśli tmux fide istnieje, nie robi prepare i tylko testuje connectivity na istniejącym środowisku. "
            "Jeśli tmux fide nie istnieje, przygotowuje venv/requests i potem testuje FIDE."
        )
    )

    parser.add_argument("--servers", default=DEFAULT_SERVERS_CSV)
    parser.add_argument("--parallel", type=int, default=3)

    parser.add_argument("--connect-timeout", type=int, default=45)
    parser.add_argument("--banner-timeout", type=int, default=90)
    parser.add_argument("--auth-timeout", type=int, default=45)
    parser.add_argument("--command-timeout", type=int, default=60)

    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=5)

    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--tries", type=int, default=2)
    parser.add_argument("--player-id", default="1503014")

    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Nie twórz venv i nie instaluj requests, nawet jeśli nie ma sesji tmux.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.parallel < 1:
        print("[ERROR] --parallel musi być >= 1")
        sys.exit(1)

    if args.connect_retries < 1:
        print("[ERROR] --connect-retries musi być >= 1")
        sys.exit(1)

    if args.tries < 1:
        print("[ERROR] --tries musi być >= 1")
        sys.exit(1)

    servers = load_servers(args.servers)
    key_path = find_first_pem()

    if not servers:
        print("[ERROR] Brak serwerów w CSV.")
        sys.exit(1)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(process_server, server, key_path, args)
            for server in servers
        ]

        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
