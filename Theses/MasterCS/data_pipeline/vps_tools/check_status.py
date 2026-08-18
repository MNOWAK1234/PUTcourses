import csv
import os
import shlex
import sys
from pathlib import Path

import paramiko


SERVERS_CSV = "servers.csv"

ROOT_USER = "root"
KEY_USER = "ubuntu"

TMUX_SESSION = "fide"
REMOTE_OUTPUT_DIR_NAME = "fide_standard_games_by_id"

# Jeśli None, bierze pierwszy plik *.pem z folderu, w którym leży ten skrypt.
DEFAULT_KEY_FILENAME = None


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_first_pem() -> str:
    if DEFAULT_KEY_FILENAME:
        p = script_dir() / DEFAULT_KEY_FILENAME
        return str(p) if p.exists() else ""

    pem_files = sorted(script_dir().glob("*.pem"))
    if not pem_files:
        return ""

    return str(pem_files[0])


def remote_dir_for_user(user: str) -> str:
    if user == "root":
        return "/root/fide_scraper"
    return f"/home/{user}/fide_scraper"


def load_servers():
    if not os.path.exists(SERVERS_CSV):
        raise FileNotFoundError(f"Nie ma pliku: {SERVERS_CSV}")

    servers = []

    with open(SERVERS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV jest pusty albo nie ma nagłówka.")

        if "host" not in reader.fieldnames or "password" not in reader.fieldnames:
            raise ValueError("CSV musi mieć kolumny: host,password")

        for row in reader:
            host = (row.get("host") or "").strip()
            password = (row.get("password") or "").strip()

            if host:
                servers.append({
                    "host": host,
                    "password": password,
                })

    return servers


def short_error(e: Exception, limit: int = 160) -> str:
    text = f"{type(e).__name__}: {str(e)}".replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    if len(text) > limit:
        return text[:limit - 3] + "..."

    return text


def new_ssh_client() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return ssh


def connect_ssh(host: str, password: str, key_path: str):
    """
    1. Jeśli password niepuste: próbuje root + password.
    2. Jeśli password puste albo nie działa: próbuje ubuntu + pierwszy *.pem z folderu skryptu.
    """
    errors = []

    if password:
        ssh = new_ssh_client()

        try:
            ssh.connect(
                hostname=host,
                username=ROOT_USER,
                password=password,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
                look_for_keys=False,
                allow_agent=False,
            )
            return ssh, ROOT_USER, "password"

        except Exception as e:
            errors.append(f"password={short_error(e)}")
            try:
                ssh.close()
            except Exception:
                pass

    if key_path:
        ssh = new_ssh_client()

        try:
            ssh.connect(
                hostname=host,
                username=KEY_USER,
                key_filename=key_path,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
                look_for_keys=False,
                allow_agent=False,
            )
            return ssh, KEY_USER, "pem"

        except Exception as e:
            errors.append(f"pem={short_error(e)}")
            try:
                ssh.close()
            except Exception:
                pass
    else:
        errors.append("pem=missing")

    raise RuntimeError("; ".join(errors) if errors else "auth failed")


def run_command(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command)

    exit_code = stdout.channel.recv_exit_status()

    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")

    return exit_code, out, err


def check_server(server, key_path: str):
    host = server["host"]
    password = server["password"]

    ssh = None

    try:
        ssh, user, auth = connect_ssh(host, password, key_path)

        remote_dir = remote_dir_for_user(user)
        run_log = f"{remote_dir}/run.log"

        command = f"""
RUNNING="no"
DONE="no"
PROGRESS=""

# Nie używamy pgrep -af scraperFaster.py, bo czasem łapie samo polecenie sprawdzające.
if ps -eo pid,ppid,stat,cmd | grep '[s]craperFaster.py' >/dev/null 2>&1; then
    RUNNING="yes"
fi

if grep -a "========== DONE ==========" {shlex.quote(run_log)} >/dev/null 2>&1; then
    DONE="yes"
fi

# Najpierw bierzemy progress z run.log, bo tqdm zapisuje tam carriage-returny (\\r).
# Po zamianie \\r -> \\n często dostajemy pełniejszą linię z errors=..., queued=...
# sed czyści przypadki typu "plaplayers:" / sklejone kawałki i zostawia ostatnie "players:".
PROGRESS=$(tr '\\r' '\\n' < {shlex.quote(run_log)} 2>/dev/null \
    | grep -a "players:" \
    | sed -E 's/^.*players:/players:/' \
    | grep -a "errors=" \
    | tail -n 1)

# Jeśli nie ma linii z errors=, weź ostatni progress z run.log.
if [ -z "$PROGRESS" ]; then
    PROGRESS=$(tr '\\r' '\\n' < {shlex.quote(run_log)} 2>/dev/null \
        | grep -a "players:" \
        | sed -E 's/^.*players:/players:/' \
        | tail -n 1)
fi

# Dopiero potem fallback do tmux. Tu też czyścimy sklejone "players:".
if [ -z "$PROGRESS" ]; then
    PROGRESS=$(tmux capture-pane -pt {shlex.quote(TMUX_SESSION)} -S -500 2>/dev/null \
        | grep -a "players:" \
        | sed -E 's/^.*players:/players:/' \
        | grep -a "errors=" \
        | tail -n 1)
fi

if [ -z "$PROGRESS" ]; then
    PROGRESS=$(tmux capture-pane -pt {shlex.quote(TMUX_SESSION)} -S -500 2>/dev/null \
        | grep -a "players:" \
        | sed -E 's/^.*players:/players:/' \
        | tail -n 1)
fi

# Jeśli nadal pusto, pokaż ostatnią sensowną linię loga.
if [ -z "$PROGRESS" ]; then
    PROGRESS=$(tail -n 30 {shlex.quote(run_log)} 2>/dev/null | grep -a -v "^$" | tail -n 1)
fi

echo "$RUNNING|$DONE|$PROGRESS"
"""

        code, out, err = run_command(ssh, command)

        line = out.strip().splitlines()[-1] if out.strip() else ""
        parts = line.split("|", 2)

        if len(parts) == 3:
            running, done, progress = parts
        else:
            running, done, progress = "unknown", "unknown", line

        print(
            f"{host} | auth={user}/{auth} | remote={remote_dir} | "
            f"running={running} | done={done} | {progress}"
        )

    except Exception as e:
        print(f"{host} | ERROR | {short_error(e, 240)}")

    finally:
        if ssh is not None:
            ssh.close()


def main():
    servers = load_servers()

    if not servers:
        print("[ERROR] Brak serwerów w CSV.")
        sys.exit(1)

    key_path = find_first_pem()

    print(f"[INFO] servers={len(servers)}")
    print(f"[INFO] pem={key_path if key_path else 'not found'}")
    print()

    for server in servers:
        check_server(server, key_path)


if __name__ == "__main__":
    main()
