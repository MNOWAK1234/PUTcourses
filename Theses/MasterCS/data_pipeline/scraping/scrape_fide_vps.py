import argparse
import atexit
import csv
import logging
import math
import os
import posixpath
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko


# Paramiko potrafi sam wypisywać wielkie tracebacki z wątku transportu,
# np. "Exception (client): Error reading SSH protocol banner".
# Te błędy i tak łapiemy w connect_ssh(), więc tu wyciszamy wewnętrzne logi Paramiko.
logging.raiseExceptions = False
for _logger_name in (
    "paramiko",
    "paramiko.transport",
    "paramiko.auth_handler",
    "paramiko.sftp",
):
    _logger = logging.getLogger(_logger_name)
    _logger.handlers.clear()
    _logger.addHandler(logging.NullHandler())
    _logger.propagate = False
    _logger.disabled = True


# =========================================
# CONFIG
# =========================================

DEFAULT_SERVERS_CSV = "servers.csv"

LOCAL_SCRAPER_FILE = Path("scraperFaster.py")
LOCAL_RATING_LIST_FILE = Path("standard_rating_list.txt")

LOCAL_FINAL_OUTPUT_DIR = Path("fide_standard_games_by_id")
LOCAL_WORK_DIR = Path("vps_archives_tmp")
LOCAL_MISSING_LIST_DIR = Path("vps_missing_lists")

REMOTE_BASE_DIR = "/root/fide_scraper"
REMOTE_OUTPUT_DIR_NAME = "fide_standard_games_by_id"
REMOTE_OUTPUT_DIR = f"{REMOTE_BASE_DIR}/{REMOTE_OUTPUT_DIR_NAME}"
TMUX_SESSION = "fide"

DEFAULT_USER = "root"
KEY_FALLBACK_USER = "ubuntu"
DEFAULT_TOTAL_PLAYERS = 549326
# Domyślnie odpalamy scraper z --workers 8.
DEFAULT_SCRAPER_WORKERS = 8

# Klucz SSH do AWS/Lightsail:
# jeśli zostawisz None, skrypt weźmie pierwszy plik *.pem z folderu, w którym leży ten skrypt.
DEFAULT_KEY_FILENAME = None

# Sprzątanie starych archiwów .tar.gz z katalogu scrapera na VPS.
# Usuwa tylko techniczne archiwa z głównego katalogu scrapera, nie rusza CSV.
CLEAN_REMOTE_TAR_GZ = True

VERBOSE = False
_MISSING_LISTS_CLEANUP_DONE = False

# SSH bywa chwilowo niestabilne na obciążonych VPS-ach, więc dajemy dłuższe timeouty i retry.
SSH_CONNECT_TIMEOUT = 45
SSH_BANNER_TIMEOUT = 90
SSH_AUTH_TIMEOUT = 45
SSH_CONNECT_RETRIES = 3
SSH_RETRY_SLEEP_SECONDS = 8


# =========================================
# LOGGING
# =========================================

def log(msg: str):
    print(msg)


def vlog(msg: str):
    if VERBOSE:
        print(msg)


# =========================================
# BASIC HELPERS
# =========================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    value = str(value).strip()
    for ch in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", "."]:
        value = value.replace(ch, "_")
    return value


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_default_key_path() -> str:
    """
    Szuka klucza SSH w tym samym folderze co ten skrypt.
    Najprościej trzymaj tam np. LightsailDefaultKey-eu-central-1.pem.
    """
    base = script_dir()

    if DEFAULT_KEY_FILENAME:
        candidate = base / DEFAULT_KEY_FILENAME
        if candidate.exists():
            return str(candidate)
        return ""

    pem_files = sorted(base.glob("*.pem"))

    if not pem_files:
        return ""

    return str(pem_files[0])


def default_remote_base_dir_for_user(user: str) -> str:
    user = (user or DEFAULT_USER).strip()
    if user == "root":
        return "/root/fide_scraper"
    return f"/home/{user}/fide_scraper"


def remote_base_dir(server_or_job: dict) -> str:
    ssh_user = server_or_job.get("ssh_user") or server_or_job.get("user") or DEFAULT_USER
    return server_or_job.get("remote_dir") or default_remote_base_dir_for_user(ssh_user)


def remote_output_dir(server_or_job: dict) -> str:
    return f"{remote_base_dir(server_or_job)}/{REMOTE_OUTPUT_DIR_NAME}"


def apt_prefix(server_or_job: dict) -> str:
    ssh_user = (server_or_job.get("ssh_user") or server_or_job.get("user") or DEFAULT_USER).strip()
    return "" if ssh_user == "root" else "sudo "


def decode_line(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


# =========================================
# LOAD SERVERS
# =========================================

def load_servers(csv_path: str):
    """
    CSV zostaje tak jak wcześniej:

      host,password
      1.2.3.4,HASLO
      3.72.75.163,

    Działanie:
    - jeśli password nie jest puste: najpierw próbuje root + password,
    - jeśli password nie działa albo jest puste: próbuje ubuntu + klucz *.pem z folderu skryptu.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nie ma pliku: {csv_path}")

    default_key_path = find_default_key_path()

    servers = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV jest pusty albo nie ma nagłówka.")

        required = {"host", "password"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError("CSV musi mieć dokładnie stary format kolumn: host,password")

        for i, row in enumerate(reader, start=1):
            host = (row.get("host") or "").strip()
            password = (row.get("password") or "").strip()

            if not host:
                continue

            servers.append({
                "index": i,
                "name": (row.get("name") or f"vps_{i}").strip(),
                "host": host,
                "user": DEFAULT_USER,
                "password": password,
                "key_path": default_key_path,
                "ssh_user": None,
                "remote_dir": None,
            })

    return servers

# =========================================
# LOCAL RATING LIST -> OFFSET TO PLAYER IDS
# =========================================

def parse_rating_list_player_ids(path: Path):
    """
    Czyta tylko ID graczy ze standard_rating_list.txt.
    To jest potrzebne do preloadu, bo shard jest po OFFSETACH listy,
    a pliki CSV mają nazwy po FIDE ID, np. 1503014.csv.
    """
    if not path.exists():
        raise FileNotFoundError(f"Nie ma pliku: {path}")

    ids = []
    seen = set()

    with open(path, "rb") as f:
        for raw in f:
            line = decode_line(raw).rstrip("\r\n")

            if not line.strip():
                continue

            if line.startswith("ID Number"):
                continue

            m = re.match(r"^\s*(\d{4,12})\s+", line)
            if not m:
                continue

            player_id = int(m.group(1))

            if player_id in seen:
                continue

            seen.add(player_id)
            ids.append(player_id)

    return ids


def get_player_ids_for_job(all_player_ids, start_offset: int, max_players: int):
    return all_player_ids[start_offset:start_offset + max_players]



def read_rating_list_records(path: Path):
    """
    Czyta oryginalny standard_rating_list.txt jako linie, a nie tylko ID.
    Dzięki temu możemy zrobić małe per-VPS listy brakujących graczy bez
    zmieniania scraperFaster.py.
    """
    if not path.exists():
        raise FileNotFoundError(f"Nie ma pliku: {path}")

    header_lines = []
    records = []
    seen = set()

    with open(path, "rb") as f:
        for raw in f:
            line = decode_line(raw).rstrip("\r\n")

            if not line.strip():
                continue

            if line.startswith("ID Number"):
                if not header_lines:
                    header_lines.append(line)
                continue

            m = re.match(r"^\s*(\d{4,12})\s+", line)
            if not m:
                continue

            player_id = m.group(1)

            if player_id in seen:
                continue

            seen.add(player_id)
            records.append({
                "player_id": player_id,
                "line": line,
            })

    return header_lines, records


def write_rating_list_records(path: Path, header_lines, records):
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if header_lines:
            f.write(header_lines[0].rstrip("\r\n") + "\n")

        for record in records:
            f.write(record["line"].rstrip("\r\n") + "\n")


def build_missing_records(rating_records, local_state, total_players: int, only_never_seen: bool = False):
    """
    Poprawna logika:

    Domyślnie do scrapowania idzie każdy gracz, który NIE jest clean.

    clean = lokalnie istnieje pid.csv i NIE istnieje pid_errors.csv.

    Czyli retry obejmuje:
    - graczy bez żadnego pliku,
    - graczy z samym pid_errors.csv,
    - graczy z pid.csv + pid_errors.csv.

    Opcja --only-never-seen włącza stare zachowanie awaryjne:
    wtedy pomijamy każdego, kto ma jakikolwiek plik lokalnie.
    """
    limited_records = rating_records[:total_players]
    local_real_ids = set(local_state["real_ids"])
    local_error_ids = set(local_state["error_ids"])

    if only_never_seen:
        done_ids = local_real_ids | local_error_ids
    else:
        done_ids = local_real_ids - local_error_ids

    missing = []
    for record in limited_records:
        pid = record["player_id"]
        if pid not in done_ids:
            missing.append(record)

    return missing


def split_even(records, parts: int):
    if parts <= 0:
        return []

    n = len(records)
    base = n // parts
    rest = n % parts
    result = []
    start = 0

    for i in range(parts):
        count = base + (1 if i < rest else 0)
        end = start + count
        result.append((start, end, records[start:end]))
        start = end

    return result


# =========================================
# LOCAL STATE FOR NEW-ONLY BACKUP
# =========================================

def classify_csv_name(name: str):
    """
    Klasyfikuje pliki CSV po samej nazwie.

    real:
        123456.csv
    error:
        123456_errors.csv
    ignored:
        *.tmp.csv
        run_errors.csv
        run_errors__*.csv
        inne dziwne CSV
    """
    name = name.strip()
    lower = name.lower()

    if not lower.endswith(".csv"):
        return "other", None

    if lower.endswith(".tmp.csv"):
        return "tmp", None

    if lower == "run_errors.csv" or lower.startswith("run_errors__"):
        return "run_errors", None

    if lower.endswith("_errors.csv"):
        pid = lower[:-len("_errors.csv")]
        return ("error", pid) if pid.isdigit() else ("other", None)

    pid = lower[:-len(".csv")]
    return ("real", pid) if pid.isdigit() else ("other", None)


def scan_local_state():
    """
    Czyta lokalny folder fide_standard_games_by_id i tworzy sety:
    - real_ids: gracze, dla których lokalnie mamy pid.csv,
    - error_ids: gracze, dla których lokalnie mamy pid_errors.csv.

    Nie czyta zawartości CSV, tylko nazwy plików, więc jest szybkie.
    """
    real_ids = set()
    error_ids = set()
    tmp_count = 0
    run_errors_count = 0
    other_count = 0

    if not LOCAL_FINAL_OUTPUT_DIR.exists():
        return {
            "real_ids": real_ids,
            "error_ids": error_ids,
            "tmp_count": tmp_count,
            "run_errors_count": run_errors_count,
            "other_count": other_count,
        }

    for path in LOCAL_FINAL_OUTPUT_DIR.glob("*.csv"):
        kind, pid = classify_csv_name(path.name)

        if kind == "real":
            real_ids.add(pid)
        elif kind == "error":
            error_ids.add(pid)
        elif kind == "tmp":
            tmp_count += 1
        elif kind == "run_errors":
            run_errors_count += 1
        else:
            other_count += 1

    return {
        "real_ids": real_ids,
        "error_ids": error_ids,
        "tmp_count": tmp_count,
        "run_errors_count": run_errors_count,
        "other_count": other_count,
    }


def decide_remote_files_to_backup(remote_names, local_real_ids, local_error_ids):
    """
    Zasada backupu new-only po poprawce:

    - zdalny pid.csv pobieramy, jeśli lokalnie NIE ma pid.csv
      ALBO lokalnie istnieje pid_errors.csv dla tego gracza.

      Dzięki temu remote pid.csv może naprawić:
      - lokalny error-only,
      - lokalny real+error.

    - zdalny pid_errors.csv pobieramy tylko jeśli lokalnie nie ma ani pid.csv,
      ani pid_errors.csv. Jeśli lokalnie już mamy error, kolejny taki sam
      error nie daje wartości; retry zrobi scraper na podstawie listy braków.

    - run_errors.csv, *.tmp.csv i inne śmieci pomijamy.
    """
    include_names = []
    remote_real_ids = set()
    remote_error_ids = set()

    skipped_existing_real = 0
    skipped_existing_error = 0
    skipped_tmp = 0
    skipped_run_errors = 0
    skipped_other = 0

    for name in remote_names:
        name = name.strip()
        if not name:
            continue

        kind, pid = classify_csv_name(name)

        if kind == "real":
            remote_real_ids.add(pid)

            # Pobierz real CSV, jeśli lokalnie brakuje reala
            # albo lokalnie jest jakikolwiek error dla tego gracza.
            # To pozwala nadpisać/naprawić error-only i real+error.
            if pid not in local_real_ids or pid in local_error_ids:
                include_names.append(name)
            else:
                skipped_existing_real += 1

        elif kind == "error":
            remote_error_ids.add(pid)

            if pid not in local_real_ids and pid not in local_error_ids:
                include_names.append(name)
            else:
                skipped_existing_error += 1

        elif kind == "tmp":
            skipped_tmp += 1

        elif kind == "run_errors":
            skipped_run_errors += 1

        else:
            skipped_other += 1

    return {
        "include_names": include_names,
        "remote_real_ids": remote_real_ids,
        "remote_error_ids": remote_error_ids,
        "skipped_existing_real": skipped_existing_real,
        "skipped_existing_error": skipped_existing_error,
        "skipped_tmp": skipped_tmp,
        "skipped_run_errors": skipped_run_errors,
        "skipped_other": skipped_other,
    }


# =========================================
# SSH HELPERS
# =========================================

def short_exception(e, max_len: int = 260) -> str:
    """
    Krótki opis wyjątku do logów, bez wielkiego tracebacka.
    """
    text = f"{type(e).__name__}: {e}"
    text = text.replace("\r", " ").replace("\n", " ").strip()

    if len(text) > max_len:
        text = text[:max_len - 3] + "..."

    return text


def connect_once(server, username: str, password: str = "", key_path: str = ""):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        kwargs = {
            "hostname": server["host"],
            "username": username,
            "timeout": SSH_CONNECT_TIMEOUT,
            "banner_timeout": SSH_BANNER_TIMEOUT,
            "auth_timeout": SSH_AUTH_TIMEOUT,
            "look_for_keys": False,
            "allow_agent": False,
        }

        if password:
            kwargs["password"] = password

        if key_path:
            kwargs["key_filename"] = key_path

        ssh.connect(**kwargs)
        return ssh

    except Exception:
        try:
            ssh.close()
        except Exception:
            pass
        raise


def connect_with_retries(server, username: str, label: str, password: str = "", key_path: str = ""):
    errors = []

    for attempt in range(1, SSH_CONNECT_RETRIES + 1):
        try:
            ssh = connect_once(
                server=server,
                username=username,
                password=password,
                key_path=key_path,
            )

            if attempt > 1:
                vlog(f"[{server['host']}] SSH {label} connected after retry {attempt}/{SSH_CONNECT_RETRIES}")

            return ssh

        except Exception as e:
            msg = short_exception(e)
            errors.append(f"{attempt}/{SSH_CONNECT_RETRIES}: {msg}")

            if attempt < SSH_CONNECT_RETRIES:
                vlog(f"[{server['host']}] SSH {label} failed {attempt}/{SSH_CONNECT_RETRIES}: {msg}; retry in {SSH_RETRY_SLEEP_SECONDS}s")
                time.sleep(SSH_RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"{label} failed after {SSH_CONNECT_RETRIES} tries: " + " | ".join(errors[-3:]))


def connect_ssh(server):
    """
    1) Jeśli w CSV jest password, próbuje: root + password.
    2) Jeśli to nie działa albo password jest puste, próbuje: ubuntu + klucz *.pem z folderu skryptu.

    Dodatkowo:
    - każda metoda ma retry, bo czasem SSH chwilowo zamyka banner,
    - Paramiko jest wyciszone, więc nie powinno wypisywać wielkich tracebacków.
    """
    errors = []

    password = server.get("password") or ""
    key_path = server.get("key_path") or ""

    if password:
        try:
            ssh = connect_with_retries(
                server=server,
                username=DEFAULT_USER,
                label="root/password",
                password=password,
            )

            server["ssh_user"] = DEFAULT_USER
            server["remote_dir"] = default_remote_base_dir_for_user(DEFAULT_USER)
            return ssh

        except Exception as e:
            errors.append(f"password={short_exception(e)}")
            vlog(f"[{server['host']}] password auth failed: {short_exception(e)}")

    if key_path:
        try:
            ssh = connect_with_retries(
                server=server,
                username=KEY_FALLBACK_USER,
                label="ubuntu/key",
                key_path=key_path,
            )

            server["ssh_user"] = KEY_FALLBACK_USER
            server["remote_dir"] = default_remote_base_dir_for_user(KEY_FALLBACK_USER)
            return ssh

        except Exception as e:
            errors.append(f"key={short_exception(e)}")
            vlog(f"[{server['host']}] key auth failed: {short_exception(e)}")
    else:
        errors.append("key=brak pliku *.pem w folderze skryptu")

    raise RuntimeError(
        f"Nie udało się połączyć z {server['host']}: "
        + " ; ".join(errors)
    )


def run_command(ssh, command: str, timeout=None, allow_fail: bool = False):
    if VERBOSE:
        print("\n" + "=" * 80)
        print("[REMOTE COMMAND]")
        print(command)
        print("=" * 80)

    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)

    exit_code = stdout.channel.recv_exit_status()

    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")

    if VERBOSE:
        if out.strip():
            print("[STDOUT]")
            print(out.strip())

        if err.strip():
            print("[STDERR]")
            print(err.strip())

        print(f"[EXIT CODE] {exit_code}")

    if exit_code != 0 and not allow_fail:
        short_err = err.strip().splitlines()[-1] if err.strip() else ""
        raise RuntimeError(f"Command failed with exit code {exit_code}: {short_err}")

    return exit_code, out, err


def upload_file(ssh, local_path: Path, remote_path: str):
    if not local_path.exists():
        raise FileNotFoundError(f"Nie ma lokalnego pliku: {local_path}")

    sftp = ssh.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def download_file(ssh, remote_path: str, local_path: Path):
    ensure_dir(local_path.parent)

    sftp = ssh.open_sftp()
    try:
        if VERBOSE:
            size = sftp.stat(remote_path).st_size
            log(f"[DOWNLOAD] {remote_path} -> {local_path} ({size / (1024 * 1024):.2f} MB)")
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def cleanup_remote_tar_gz(ssh, job, reason: str = ""):
    """
    Usuwa techniczne archiwa .tar.gz z głównego katalogu scrapera na VPS.

    Nie usuwa CSV i nie wchodzi do fide_standard_games_by_id.
    """
    if not CLEAN_REMOTE_TAR_GZ:
        return 0

    rdir = remote_base_dir(job)

    code, out, err = run_command(
        ssh,
        (
            f"if [ -d {shlex.quote(rdir)} ]; then "
            f"COUNT=$(find {shlex.quote(rdir)} -maxdepth 1 -type f -name '*.tar.gz' | wc -l); "
            f"find {shlex.quote(rdir)} -maxdepth 1 -type f -name '*.tar.gz' -delete; "
            f"echo TAR_GZ_DELETED=$COUNT; "
            f"else echo TAR_GZ_DELETED=0; fi"
        ),
        timeout=120,
        allow_fail=True,
    )

    last_line = out.strip().splitlines()[-1] if out.strip() else "TAR_GZ_DELETED=0"

    try:
        deleted = int(last_line.split("=", 1)[1])
    except Exception:
        deleted = 0

    if deleted:
        log(f"[{job['host']}] cleaned old remote tar.gz={deleted}" + (f" ({reason})" if reason else ""))

    return deleted



# =========================================
# PREFLIGHT + MISSING LIST PLAN
# =========================================

def preflight_server(server):
    ssh = None

    try:
        ssh = connect_ssh(server)
        run_command(ssh, "echo ok", timeout=30)
        return True, server, "ok"

    except Exception as e:
        return False, server, repr(e)

    finally:
        if ssh is not None:
            ssh.close()


def build_missing_shard_plan(working_servers, missing_records):
    """
    Dzieli wyłącznie brakujących graczy między maszyny, które przeszły test FIDE.
    Nie ma już shardowania po globalnym offsecie oryginalnej listy.
    """
    if not working_servers or not missing_records:
        return []

    plan = []
    splits = split_even(missing_records, len(working_servers))

    for server, (missing_start, missing_end, records) in zip(working_servers, splits):
        if not records:
            continue

        job = dict(server)
        job["missing_start"] = missing_start
        job["missing_end_exclusive"] = missing_end
        job["max_players"] = len(records)
        job["records"] = records
        job["start_offset"] = 0
        job["end_exclusive"] = len(records)
        job["first_player_id"] = records[0]["player_id"]
        job["last_player_id"] = records[-1]["player_id"]
        job["local_rating_list_path"] = None
        job["local_shard_info_path"] = None
        plan.append(job)

    return plan


def write_missing_lists(header_lines, missing_records, plan):
    ensure_dir(LOCAL_MISSING_LIST_DIR)

    all_path = LOCAL_MISSING_LIST_DIR / "missing_all_standard_rating_list.txt"
    write_rating_list_records(all_path, header_lines, missing_records)

    for i, job in enumerate(plan, start=1):
        host_safe = safe_name(job["host"])
        list_path = LOCAL_MISSING_LIST_DIR / f"standard_rating_list_missing_{i:03d}_{host_safe}.txt"
        info_path = LOCAL_MISSING_LIST_DIR / f"shard_info_{i:03d}_{host_safe}.txt"

        write_rating_list_records(list_path, header_lines, job["records"])

        with open(info_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"host={job['host']}\n")
            f.write(f"name={job.get('name', '')}\n")
            f.write(f"remote_dir={remote_base_dir(job)}\n")
            f.write(f"missing_start={job['missing_start']}\n")
            f.write(f"missing_end_exclusive={job['missing_end_exclusive']}\n")
            f.write(f"count={job['max_players']}\n")
            f.write(f"first_player_id={job['first_player_id']}\n")
            f.write(f"last_player_id={job['last_player_id']}\n")
            f.write("scraper_start_offset=0\n")
            f.write("scraper_max_players=none\n")

        job["local_rating_list_path"] = list_path
        job["local_shard_info_path"] = info_path

    return all_path


def cleanup_local_missing_lists(keep: bool):
    """
    Sprząta lokalne tymczasowe listy brakujących graczy.

    Domyślnie usuwamy folder vps_missing_lists na końcu działania skryptu,
    bo te pliki są tylko techniczne. Jeśli chcesz je zostawić do debugowania,
    odpal skrypt z --keep-missing-lists.

    Funkcja jest idempotentna, bo może zostać wywołana normalnie w main()
    oraz awaryjnie przez atexit przy sys.exit albo błędzie.
    """
    global _MISSING_LISTS_CLEANUP_DONE

    if _MISSING_LISTS_CLEANUP_DONE:
        return

    _MISSING_LISTS_CLEANUP_DONE = True

    if keep:
        if LOCAL_MISSING_LIST_DIR.exists():
            log(f"[INFO] keep missing lists: {LOCAL_MISSING_LIST_DIR}")
        return

    if LOCAL_MISSING_LIST_DIR.exists():
        shutil.rmtree(LOCAL_MISSING_LIST_DIR, ignore_errors=True)
        log(f"[INFO] cleaned local missing lists: {LOCAL_MISSING_LIST_DIR}")


def save_plan(plan, path="shard_plan.csv"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "name",
            "host",
            "user",
            "ssh_user",
            "remote_dir",
            "missing_start",
            "missing_end_exclusive",
            "count",
            "first_player_id",
            "last_player_id",
            "local_rating_list_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for job in plan:
            writer.writerow({
                "name": job.get("name", ""),
                "host": job["host"],
                "user": job.get("user", DEFAULT_USER),
                "ssh_user": job.get("ssh_user") or job.get("user") or DEFAULT_USER,
                "remote_dir": remote_base_dir(job),
                "missing_start": job["missing_start"],
                "missing_end_exclusive": job["missing_end_exclusive"],
                "count": job["max_players"],
                "first_player_id": job["first_player_id"],
                "last_player_id": job["last_player_id"],
                "local_rating_list_path": str(job.get("local_rating_list_path") or ""),
            })

# =========================================
# REMOTE BACKUP -> LOCAL MERGE
# =========================================

def is_safe_tar_member(base_dir: Path, member_name: str) -> bool:
    target_path = (base_dir / member_name).resolve()
    base_resolved = base_dir.resolve()
    return str(target_path).startswith(str(base_resolved))


def safe_extract_tar(tar_path: Path, extract_to: Path):
    ensure_dir(extract_to)

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not is_safe_tar_member(extract_to, member.name):
                raise RuntimeError(f"Unsafe tar member: {member.name}")

        try:
            tar.extractall(extract_to, filter="data")
        except TypeError:
            # Python < 3.12 nie ma parametru filter.
            tar.extractall(extract_to)


def copy_csv_to_final(src: Path, final_dir: Path, prefix: str) -> str:
    """
    Kopiuje CSV z backupu VPS do lokalnego fide_standard_games_by_id.

    Zasady:
    - *.tmp.csv pomijamy,
    - run_errors.csv / run_errors__*.csv pomijamy,
    - normalne pliki graczy, np. 123456.csv albo 123456_errors.csv:
        jeśli lokalnie ich nie ma -> kopiujemy,
        jeśli lokalnie istnieją -> NADPISUJEMY wersją z VPS.

    Przy backupie new-only nadpisania powinny być rzadkie, ale zostawiamy
    to jako bezpieczne zachowanie, jeśli lokalny stan zmienił się między
    skanowaniem a merge.
    """
    ensure_dir(final_dir)

    kind, pid = classify_csv_name(src.name)

    if kind == "tmp":
        return "skipped_tmp"

    if kind == "run_errors":
        return "skipped_run_errors"

    if kind not in ("real", "error"):
        return "skipped_other"

    dst = final_dir / src.name

    if not dst.exists():
        shutil.copy2(src, dst)
        return "copied_new"

    shutil.copy2(src, dst)
    return "overwritten"


def get_player_id_from_game_csv_name(name: str):
    kind, pid = classify_csv_name(name)
    return pid if kind == "real" else None


def get_player_id_from_error_csv_name(name: str):
    kind, pid = classify_csv_name(name)
    return pid if kind == "error" else None


def merge_archive_into_final(archive_path: Path, prefix: str, remote_error_player_ids=None):
    extract_to = LOCAL_WORK_DIR / f"extracted_backup_{prefix}"

    if extract_to.exists():
        shutil.rmtree(extract_to)

    safe_extract_tar(archive_path, extract_to)

    copied_new = 0
    overwritten = 0
    run_errors = 0
    skipped = 0
    tmp_skipped = 0
    stale_errors_deleted = 0

    csv_files = sorted(extract_to.rglob("*.csv"))

    # W trybie new-only archiwum nie musi zawierać wszystkich zdalnych _errors.csv,
    # dlatego lista remote_error_player_ids powinna pochodzić z pełnej listy nazw
    # na VPS. Jeśli jej nie podano, fallbackujemy do archiwum.
    if remote_error_player_ids is None:
        remote_error_player_ids = set()

        for csv_path in csv_files:
            pid = get_player_id_from_error_csv_name(csv_path.name)
            if pid is not None:
                remote_error_player_ids.add(pid)

    remote_success_player_ids = set()

    for csv_path in csv_files:
        status = copy_csv_to_final(csv_path, LOCAL_FINAL_OUTPUT_DIR, prefix)

        if status == "skipped_tmp":
            tmp_skipped += 1
        elif status == "skipped_run_errors":
            run_errors += 1
        elif status == "copied_new":
            copied_new += 1
        elif status == "overwritten":
            overwritten += 1
        elif status.startswith("skipped"):
            skipped += 1
        else:
            copied_new += 1

        pid = get_player_id_from_game_csv_name(csv_path.name)
        if pid is not None:
            remote_success_player_ids.add(pid)

    for pid in remote_success_player_ids:
        if pid in remote_error_player_ids:
            continue

        local_error_path = LOCAL_FINAL_OUTPUT_DIR / f"{pid}_errors.csv"

        if local_error_path.exists():
            local_error_path.unlink()
            stale_errors_deleted += 1

    shutil.rmtree(extract_to, ignore_errors=True)

    return {
        "csv_found": len(csv_files),
        "copied_new": copied_new,
        "overwritten": overwritten,
        "run_errors": run_errors,
        "skipped": skipped,
        "tmp_skipped": tmp_skipped,
        "stale_errors_deleted": stale_errors_deleted,
    }


def stop_existing_scraper(ssh):
    command = (
        f"tmux send-keys -t {shlex.quote(TMUX_SESSION)} C-c 2>/dev/null || true; "
        f"sleep 3; "
        f"tmux kill-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null || true; "
        f"pkill -INT -f scraperFaster.py 2>/dev/null || true; "
        f"sleep 2; "
        f"pkill -KILL -f scraperFaster.py 2>/dev/null || true; "
        f"echo stopped"
    )

    run_command(ssh, command, timeout=60, allow_fail=True)


def remote_fide_exists(ssh, job):
    rdir = remote_base_dir(job)

    code, out, err = run_command(
        ssh,
        f"test -d {shlex.quote(rdir)} && echo yes || echo no",
        timeout=30,
        allow_fail=True,
    )

    return out.strip().splitlines()[-1] == "yes"


def remote_output_exists(ssh, job):
    out_dir = remote_output_dir(job)

    code, out, err = run_command(
        ssh,
        f"test -d {shlex.quote(out_dir)} && echo yes || echo no",
        timeout=30,
        allow_fail=True,
    )

    return out.strip().splitlines()[-1] == "yes"


def list_remote_csv_names(ssh, job):
    out_dir = remote_output_dir(job)

    command = (
        f"if [ -d {shlex.quote(out_dir)} ]; then "
        f"find {shlex.quote(out_dir)} -maxdepth 1 -type f -name '*.csv' -printf '%f\\n' 2>/dev/null; "
        f"fi"
    )

    code, out, err = run_command(ssh, command, timeout=180, allow_fail=True)

    names = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            names.append(line)

    return names


def upload_include_list(ssh, job, archive_name: str, include_names):
    """
    Wysyła na VPS jeden mały plik tekstowy z listą plików, które tar ma spakować.
    Każda linia ma postać:
        fide_standard_games_by_id/123456.csv
    """
    rdir = remote_base_dir(job)
    include_name = archive_name.replace(".tar.gz", "_include.txt")

    ensure_dir(LOCAL_WORK_DIR)

    local_include_path = LOCAL_WORK_DIR / include_name
    remote_include_path = posixpath.join(rdir, include_name)

    with open(local_include_path, "w", encoding="utf-8", newline="\n") as f:
        for name in include_names:
            if "/" in name or "\\" in name or name in ("", ".", ".."):
                continue
            f.write(f"{REMOTE_OUTPUT_DIR_NAME}/{name}\n")

    upload_file(ssh, local_include_path, remote_include_path)

    try:
        local_include_path.unlink()
    except OSError:
        pass

    return remote_include_path


def backup_existing_remote_results(ssh, job, local_state):
    """
    Jeśli na VPS są stare wyniki, zatrzymuje scraper, robi backup tylko nowych/
    użytecznych plików względem lokalnego fide_standard_games_by_id, pobiera
    archiwum i scala lokalnie.

    Setup/start dalej działa normalnie. Zmienia się tylko backup:
    zamiast pakować cały zdalny output, pakujemy tylko include-list.
    """
    host = job["host"]
    prefix = safe_name(f"{job['name']}_{host}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_name = f"backup_new_only_{prefix}_{timestamp}.tar.gz"

    rdir = remote_base_dir(job)

    cleanup_remote_tar_gz(ssh, job, reason="before backup")

    remote_archive_path = posixpath.join(rdir, archive_name)
    local_archive_path = LOCAL_WORK_DIR / archive_name
    remote_include_path = None

    if not remote_fide_exists(ssh, job):
        return True, None

    stop_existing_scraper(ssh)

    if not remote_output_exists(ssh, job):
        return True, None

    remote_names = list_remote_csv_names(ssh, job)

    decision = decide_remote_files_to_backup(
        remote_names=remote_names,
        local_real_ids=local_state["real_ids"],
        local_error_ids=local_state["error_ids"],
    )

    include_names = decision["include_names"]

    log(
        f"[{host}] backup old remote csv={len(remote_names)} "
        f"new_to_backup={len(include_names)} "
        f"skip_real={decision['skipped_existing_real']} "
        f"skip_error={decision['skipped_existing_error']}"
    )

    if not include_names:
        cleanup_remote_tar_gz(ssh, job, reason="nothing new")
        stats = {
            "csv_found": 0,
            "copied_new": 0,
            "overwritten": 0,
            "run_errors": 0,
            "skipped": 0,
            "tmp_skipped": decision["skipped_tmp"],
            "stale_errors_deleted": 0,
            "remote_csv": len(remote_names),
            "new_to_backup": 0,
        }

        log(
            f"[{host}] backup merged | remote_csv={stats['remote_csv']} "
            f"new_to_backup=0 found=0 new=0 overwritten=0 "
            f"stale_errors_deleted=0 skipped=0"
        )

        return True, stats

    remote_include_path = upload_include_list(ssh, job, archive_name, include_names)

    tar_command = (
        f"cd {shlex.quote(rdir)} && "
        f"tar --warning=no-file-changed "
        f"--ignore-failed-read "
        f"-czf {shlex.quote(archive_name)} "
        f"-T {shlex.quote(remote_include_path)}"
    )

    log(f"[{host}] packing new-only files={len(include_names)}...")
    run_command(ssh, tar_command, timeout=None)

    log(f"[{host}] downloading new-only archive...")
    download_file(ssh, remote_archive_path, local_archive_path)

    stats = merge_archive_into_final(
        archive_path=local_archive_path,
        prefix=prefix,
        remote_error_player_ids=decision["remote_error_ids"],
    )

    stats["remote_csv"] = len(remote_names)
    stats["new_to_backup"] = len(include_names)

    run_command(
        ssh,
        f"rm -f {shlex.quote(remote_archive_path)} {shlex.quote(remote_include_path)}",
        timeout=60,
        allow_fail=True,
    )

    cleanup_remote_tar_gz(ssh, job, reason="after backup")

    try:
        local_archive_path.unlink()
    except OSError:
        pass

    log(
        f"[{host}] backup merged | remote_csv={stats['remote_csv']} "
        f"new_to_backup={stats['new_to_backup']} "
        f"found={stats['csv_found']} "
        f"new={stats['copied_new']} overwritten={stats['overwritten']} "
        f"stale_errors_deleted={stats['stale_errors_deleted']} "
        f"skipped={stats['skipped']}"
    )

    return True, stats

def clean_remote_output_after_successful_backup(ssh, job):
    """
    Domyślny tryb: po udanym backupie usuwa CAŁY zdalny output folder
    fide_standard_games_by_id, ale nie usuwa całego katalogu roboczego.

    Czyli:
    - backup najpierw zapisuje stare CSV lokalnie,
    - potem na VPS znika cały stary output,
    - później standardowy setup + preload wgrają z powrotem tylko pliki
      potrzebne dla nowego zakresu tej maszyny.

    Dodatkowo kasujemy run.log, żeby nowa sesja tmux miała świeży log.
    """
    rdir = remote_base_dir(job)
    out_dir = remote_output_dir(job)

    code, out, err = run_command(
        ssh,
        (
            f"mkdir -p {shlex.quote(rdir)} && "
            f"rm -rf {shlex.quote(out_dir)} && "
            f"mkdir -p {shlex.quote(out_dir)} && "
            f"rm -f {shlex.quote(rdir)}/run.log && "
            f"CSV_LEFT=$(find {shlex.quote(out_dir)} -name '*.csv' 2>/dev/null | wc -l); "
            f"echo CSV_LEFT=$CSV_LEFT"
        ),
        timeout=180,
    )

    last_line = out.strip().splitlines()[-1] if out.strip() else ""

    if last_line != "CSV_LEFT=0":
        raise RuntimeError(f"Remote output cleanup check failed for {out_dir}: {last_line!r}; stderr={err.strip()!r}")

    return "OUTPUT_CSV_LEFT=0"


def clean_remote_full_reset_after_successful_backup(ssh, job):
    """
    Ostrzejszy tryb opcjonalny: po udanym backupie usuwa CAŁY zdalny katalog
    roboczy scrapera i tworzy go od nowa.
    """
    rdir = remote_base_dir(job)

    code, out, err = run_command(
        ssh,
        (
            f"rm -rf {shlex.quote(rdir)} && "
            f"mkdir -p {shlex.quote(rdir)} && "
            f"CSV_LEFT=$(find {shlex.quote(rdir)} -name '*.csv' 2>/dev/null | wc -l); "
            f"echo CSV_LEFT=$CSV_LEFT"
        ),
        timeout=180,
    )

    last_line = out.strip().splitlines()[-1] if out.strip() else ""

    if last_line != "CSV_LEFT=0":
        raise RuntimeError(f"Remote full cleanup check failed for {rdir}: {last_line!r}; stderr={err.strip()!r}")

    return "FULL_CSV_LEFT=0"


def clean_remote_after_successful_backup(ssh, job, all_player_ids, cleanup_mode: str):
    """
    Wołane dopiero po udanym backup_existing_remote_results().

    cleanup_mode:
    - output-reset: domyślnie, usuwa cały zdalny output folder z CSV,
    - full-reset: usuwa cały zdalny katalog roboczy scrapera,
    - none: nic nie usuwa.
    """
    if cleanup_mode == "none":
        return "SKIPPED"

    if cleanup_mode == "output-reset":
        return clean_remote_output_after_successful_backup(ssh, job)

    if cleanup_mode == "full-reset":
        return clean_remote_full_reset_after_successful_backup(ssh, job)

    raise ValueError(f"Unknown cleanup mode: {cleanup_mode}")



# =========================================
# SETUP + FIDE TEST + START
# =========================================

def setup_vps(ssh, job):
    rdir = remote_base_dir(job)
    sudo = apt_prefix(job)

    # Bez apt upgrade - mniej ryzyka, mniej czasu, mniej przebudowy systemu.
    run_command(
        ssh,
        f"export DEBIAN_FRONTEND=noninteractive && {sudo}apt update -y",
        timeout=None,
    )

    run_command(
        ssh,
        f"export DEBIAN_FRONTEND=noninteractive && {sudo}apt install -y python3 python3-pip python3-venv tmux htop tar",
        timeout=None,
    )

    run_command(
        ssh,
        f"mkdir -p {shlex.quote(rdir)} {shlex.quote(rdir)}/{REMOTE_OUTPUT_DIR_NAME}",
        timeout=60,
    )

    run_command(
        ssh,
        (
            f"cd {shlex.quote(rdir)} && "
            f"if [ ! -d venv ]; then python3 -m venv venv; fi && "
            f". venv/bin/activate && "
            f"python3 -m pip install --upgrade pip && "
            f"pip install requests beautifulsoup4 lxml tqdm"
        ),
        timeout=None,
    )


def check_fide_requests(ssh, job, player_id: str, tries: int, request_timeout: int):
    rdir = remote_base_dir(job)

    py_code = r'''
import re
import time
import requests
from urllib.parse import unquote

BASE = "https://ratings.fide.com/"
PLAYER_ID = "__PLAYER_ID__"
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

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
    })

    try:
        t = time.time()
        r = s.get(BASE, timeout=REQUEST_TIMEOUT)
        result["home_status"] = str(r.status_code)
        result["home_seconds"] = f"{time.time() - t:.2f}"
        result["home"] = (200 <= r.status_code < 500 and len(r.text) > 0)
    except Exception as e:
        result["error"] = "home:" + repr(e)[:180]
        return result

    try:
        t = time.time()
        r = s.post(
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
        result["post_status"] = str(r.status_code)
        result["post_seconds"] = f"{time.time() - t:.2f}"

        html = r.text or ""
        result["post"] = (200 <= r.status_code < 500 and "calculations.phtml" in html and len(html) > 0)

        m = re.search(
            r'calculations\.phtml\?id_number=' + re.escape(PLAYER_ID) + r'&period=([^&"\']+)&rating=0',
            html,
        )

        if not m:
            m = re.search(r'period=([^&"\']+)', html)

        if not m:
            result["error"] = "post:no_period_link"
            return result

        period = unquote(m.group(1))
        result["period"] = period

    except Exception as e:
        result["error"] = "post:" + repr(e)[:180]
        return result

    try:
        t = time.time()
        r = s.get(
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
        result["calc_status"] = str(r.status_code)
        result["calc_seconds"] = f"{time.time() - t:.2f}"

        text = r.text or ""
        result["calc"] = (200 <= r.status_code < 500 and len(text) > 0)

        if not result["calc"]:
            result["error"] = "calc:empty_or_bad_status"

    except Exception as e:
        result["error"] = "calc:" + repr(e)[:180]
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
        .replace("__PLAYER_ID__", str(player_id))
        .replace("__TRIES__", str(tries))
        .replace("__REQUEST_TIMEOUT__", str(request_timeout))
    )

    command = (
        f"cd {shlex.quote(rdir)}\n"
        ". venv/bin/activate\n"
        "python - <<'REMOTE_PY'\n"
        f"{py_code}\n"
        "REMOTE_PY"
    ).strip()

    code, out, err = run_command(ssh, command, timeout=(tries * (request_timeout * 3 + 20) + 60), allow_fail=True)

    result = {
        "status": "FIDE ERROR",
        "home": "0/0",
        "post": "0/0",
        "calc": "0/0",
        "first_bad": "unknown",
        "details": (err.strip() or out.strip() or f"code={code}")[:300],
    }

    if code != 0:
        return result

    result_line = ""
    for line in out.splitlines():
        if line.startswith("RESULT "):
            result_line = line
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
    result["details"] = " ".join(f"{k}={parts.get(k, '-') }" for k in detail_keys)

    expected = f"{tries}/{tries}"
    if result["home"] == expected and result["post"] == expected and result["calc"] == expected:
        result["status"] = "FIDE OK"
    else:
        result["status"] = "FIDE NOT OK"

    return result


def upload_scraper_and_missing_list(ssh, job):
    rdir = remote_base_dir(job)

    if not job.get("local_rating_list_path"):
        raise RuntimeError("Job has no local_rating_list_path")

    upload_file(ssh, LOCAL_SCRAPER_FILE, f"{rdir}/scraperFaster.py")
    upload_file(ssh, Path(job["local_rating_list_path"]), f"{rdir}/standard_rating_list.txt")

    if job.get("local_shard_info_path"):
        upload_file(ssh, Path(job["local_shard_info_path"]), f"{rdir}/shard_info.txt")


def start_scraper_in_tmux(ssh, job, workers):
    rdir = remote_base_dir(job)

    scraper_command = (
        f"cd {shlex.quote(rdir)} && "
        f". venv/bin/activate && "
        f"python3 scraperFaster.py "
        f"--rating-list standard_rating_list.txt "
        f"--start-offset 0 "
        f"--max-players none "
        f"--workers {workers} "
        f"2>&1 | tee -a run.log"
    )

    tmux_command = (
        f"tmux kill-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null || true; "
        f"tmux new-session -d -s {shlex.quote(TMUX_SESSION)} "
        f"{shlex.quote('bash -lc ' + shlex.quote(scraper_command))}; "
        f"tmux ls"
    )

    run_command(ssh, tmux_command, timeout=60)


def quick_status(ssh, job):
    rdir = remote_base_dir(job)

    command = (
        f"sleep 2; "
        f"echo RUNNING=$(pgrep -af scraperFaster.py >/dev/null 2>&1 && echo yes || echo no); "
        f"tail -n 12 {shlex.quote(rdir)}/run.log 2>/dev/null | tail -n 1 || true"
    )

    code, out, err = run_command(ssh, command, timeout=60, allow_fail=True)
    return out.strip()


def backup_server(server):
    host = server["host"]
    ssh = None

    try:
        local_state = scan_local_state()
        log(f"[{host}] backup phase | local real={len(local_state['real_ids'])} error={len(local_state['error_ids'])}")

        ssh = connect_ssh(server)
        backup_ok, backup_stats = backup_existing_remote_results(ssh, server, local_state)

        if not backup_ok:
            raise RuntimeError("Backup old remote results failed.")

        return True, server, None

    except Exception as e:
        log(f"[{host}] BACKUP/SSH ERROR | {repr(e)}")
        return False, server, repr(e)

    finally:
        if ssh is not None:
            ssh.close()


def setup_and_test_server(server, args):
    host = server["host"]
    ssh = None

    try:
        ssh = connect_ssh(server)

        log(f"[{host}] setup venv/deps...")
        setup_vps(ssh, server)

        if args.skip_fide_test:
            result = {
                "status": "SKIPPED",
                "home": "-",
                "post": "-",
                "calc": "-",
                "first_bad": "-",
                "details": "--skip-fide-test",
            }
            log(f"[{host}] FIDE TEST SKIPPED")
            return True, server, result, None

        log(f"[{host}] FIDE connectivity test...")
        result = check_fide_requests(
            ssh=ssh,
            job=server,
            player_id=args.fide_test_player_id,
            tries=args.fide_tries,
            request_timeout=args.request_timeout,
        )

        ok = result["status"] == "FIDE OK"
        log(
            f"[{host}] {result['status']} "
            f"home={result['home']} post={result['post']} calc={result['calc']} "
            f"first_bad={result['first_bad']} | {result['details']}"
        )

        return ok, server, result, None if ok else result["details"]

    except Exception as e:
        log(f"[{host}] SETUP/TEST ERROR | {repr(e)}")
        result = {
            "status": "SETUP/TEST ERROR",
            "home": "0/0",
            "post": "0/0",
            "calc": "0/0",
            "first_bad": "setup",
            "details": short_exception(e),
        }
        return False, server, result, repr(e)

    finally:
        if ssh is not None:
            ssh.close()


def process_start_job(job, workers, cleanup_mode):
    host = job["host"]
    ssh = None

    try:
        log(
            f"[{host}] start missing_index={job['missing_start']}-{job['missing_end_exclusive'] - 1} "
            f"count={job['max_players']} ids={job['first_player_id']}..{job['last_player_id']} "
            f"workers={workers} remote={remote_base_dir(job)}"
        )

        ssh = connect_ssh(job)

        log(f"[{host}] clear remote output ({cleanup_mode})...")
        cleanup_status = clean_remote_after_successful_backup(ssh, job, [], cleanup_mode)
        log(f"[{host}] clear OK | {cleanup_status}")

        log(f"[{host}] upload scraper + missing list...")
        upload_scraper_and_missing_list(ssh, job)

        start_scraper_in_tmux(ssh, job, workers)

        status = quick_status(ssh, job)
        log(f"[{host}] OK | {status}")

        return True, host, None

    except Exception as e:
        log(f"[{host}] START ERROR | {repr(e)}")
        return False, host, repr(e)

    finally:
        if ssh is not None:
            ssh.close()


# =========================================
# MAIN
# =========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Backup new-only z VPS, lokalne wyliczenie brakujących graczy, setup/test FIDE i start tylko na działających maszynach."
    )

    parser.add_argument("--servers", default=DEFAULT_SERVERS_CSV, help="CSV ze starym formatem host,password. Domyślnie: servers.csv")
    parser.add_argument("--total-players", type=int, default=DEFAULT_TOTAL_PLAYERS, help="Ile pierwszych graczy z rating listy brać pod uwagę.")
    parser.add_argument("--parallel", type=int, default=5, help="Ile VPS-ów robić równolegle przy setup/test/start.")
    parser.add_argument("--workers", type=int, default=DEFAULT_SCRAPER_WORKERS, help=f"Ile workerów przekazać do scraperFaster.py. Domyślnie: {DEFAULT_SCRAPER_WORKERS}")
    parser.add_argument("--remote-cleanup", choices=["output-reset", "full-reset", "none"], default="output-reset", help="Co usuwać na VPS przed startem. Domyślnie output-reset.")
    parser.add_argument("--retry-errors", action="store_true", help="DEPRECATED/no-op: retry errorów jest teraz domyślne.")
    parser.add_argument("--only-never-seen", action="store_true", help="Stare zachowanie awaryjne: scrapuj tylko graczy, którzy lokalnie nie mają ani pid.csv, ani pid_errors.csv.")
    parser.add_argument("--keep-missing-lists", action="store_true", help="Nie usuwaj lokalnego folderu vps_missing_lists po normalnym zakończeniu skryptu.")
    parser.add_argument("--skip-fide-test", action="store_true", help="Nie rób testu FIDE, startuj na wszystkich maszynach po setupie.")
    parser.add_argument("--fide-test-player-id", default="1503014", help="FIDE ID używane do real-flow testu connectivity. Domyślnie: 1503014")
    parser.add_argument("--fide-tries", type=int, default=2, help="Ile pełnych prób FIDE zrobić na maszynę. Domyślnie: 2")
    parser.add_argument("--request-timeout", type=int, default=30, help="Timeout HTTP requestów do FIDE w teście. Domyślnie: 30")
    parser.add_argument("--verbose", action="store_true", help="Pokaż pełne komendy i stdout/stderr.")
    parser.add_argument("--no-clean-remote-tar-gz", action="store_true", help="Nie usuwaj starych *.tar.gz z katalogu scrapera na VPS.")
    parser.add_argument("--connect-retries", type=int, default=SSH_CONNECT_RETRIES, help=f"Ile razy próbować SSH dla jednej metody auth. Domyślnie: {SSH_CONNECT_RETRIES}")
    parser.add_argument("--retry-sleep", type=int, default=SSH_RETRY_SLEEP_SECONDS, help=f"Ile sekund czekać między próbami SSH. Domyślnie: {SSH_RETRY_SLEEP_SECONDS}")
    parser.add_argument("--ssh-timeout", type=int, default=SSH_CONNECT_TIMEOUT, help=f"Timeout TCP SSH w sekundach. Domyślnie: {SSH_CONNECT_TIMEOUT}")
    parser.add_argument("--banner-timeout", type=int, default=SSH_BANNER_TIMEOUT, help=f"Timeout na SSH banner w sekundach. Domyślnie: {SSH_BANNER_TIMEOUT}")
    parser.add_argument("--auth-timeout", type=int, default=SSH_AUTH_TIMEOUT, help=f"Timeout auth SSH w sekundach. Domyślnie: {SSH_AUTH_TIMEOUT}")

    return parser.parse_args()


def main():
    global VERBOSE, CLEAN_REMOTE_TAR_GZ
    global SSH_CONNECT_RETRIES, SSH_RETRY_SLEEP_SECONDS, SSH_CONNECT_TIMEOUT, SSH_BANNER_TIMEOUT, SSH_AUTH_TIMEOUT

    args = parse_args()
    atexit.register(lambda: cleanup_local_missing_lists(args.keep_missing_lists))

    VERBOSE = args.verbose
    CLEAN_REMOTE_TAR_GZ = not args.no_clean_remote_tar_gz

    SSH_CONNECT_RETRIES = max(1, args.connect_retries)
    SSH_RETRY_SLEEP_SECONDS = max(0, args.retry_sleep)
    SSH_CONNECT_TIMEOUT = max(5, args.ssh_timeout)
    SSH_BANNER_TIMEOUT = max(5, args.banner_timeout)
    SSH_AUTH_TIMEOUT = max(5, args.auth_timeout)

    if args.retry_errors:
        log("[INFO] --retry-errors jest już domyślne; flaga zostawiona tylko dla kompatybilności.")

    if args.workers < 1:
        print("[ERROR] --workers musi być >= 1")
        sys.exit(1)

    if args.parallel < 1:
        print("[ERROR] --parallel musi być >= 1")
        sys.exit(1)

    if args.fide_tries < 1:
        print("[ERROR] --fide-tries musi być >= 1")
        sys.exit(1)

    ensure_dir(LOCAL_FINAL_OUTPUT_DIR)
    ensure_dir(LOCAL_WORK_DIR)
    ensure_dir(LOCAL_MISSING_LIST_DIR)

    if not LOCAL_SCRAPER_FILE.exists():
        print(f"[ERROR] Brak pliku: {LOCAL_SCRAPER_FILE}")
        sys.exit(1)

    if not LOCAL_RATING_LIST_FILE.exists():
        print(f"[ERROR] Brak pliku: {LOCAL_RATING_LIST_FILE}")
        sys.exit(1)

    servers = load_servers(args.servers)

    if not servers:
        print("[ERROR] Brak serwerów w CSV.")
        sys.exit(1)

    log(f"[INFO] servers in CSV: {len(servers)}")
    log("[INFO] reading local rating list...")
    header_lines, rating_records = read_rating_list_records(LOCAL_RATING_LIST_FILE)
    log(f"[INFO] player records in rating list: {len(rating_records)}")

    if args.total_players > len(rating_records):
        log(f"[WARN] total_players={args.total_players}, ale lista ma tylko {len(rating_records)} ID.")
        log(f"[WARN] Używam total_players={len(rating_records)}.")
        args.total_players = len(rating_records)

    log("")
    log("========== PHASE 1: BACKUP EACH MACHINE NEW-ONLY ==========")
    backup_ok_servers = []
    backup_failed = []

    # Backup robimy sekwencyjnie, bo po każdym merge zmienia się lokalny stan.
    # To ogranicza duplikaty w kolejnych backupach.
    for server in servers:
        ok, updated_server, err = backup_server(server)
        if ok:
            backup_ok_servers.append(updated_server)
        else:
            backup_failed.append((updated_server, err))

    if not backup_ok_servers:
        print("[ERROR] Nie udało się połączyć/zbackupować żadnej maszyny.")
        sys.exit(1)

    local_state = scan_local_state()
    local_real = len(local_state["real_ids"])
    local_error = len(local_state["error_ids"])
    local_any = len(local_state["real_ids"] | local_state["error_ids"])

    missing_records = build_missing_records(
        rating_records=rating_records,
        local_state=local_state,
        total_players=args.total_players,
        only_never_seen=args.only_never_seen,
    )

    # Zapisz globalną listę braków już teraz, nawet zanim wiemy, które maszyny przejdą FIDE.
    all_missing_path = LOCAL_MISSING_LIST_DIR / "missing_all_standard_rating_list.txt"
    write_rating_list_records(all_missing_path, header_lines, missing_records)

    log("")
    log("========== LOCAL STATE AFTER BACKUP ==========")
    log(f"total considered players: {args.total_players}")
    log(f"local real players:      {local_real}")
    log(f"local error players:     {local_error}")
    log(f"local any players:       {local_any}")
    clean_done = len(local_state["real_ids"] - local_state["error_ids"])
    log(f"local clean players:     {clean_done}")
    log(f"missing/retry to assign: {len(missing_records)}")
    log(f"retry errors:            yes (default)")
    log(f"only never seen mode:    {args.only_never_seen}")
    log(f"missing list:            {all_missing_path}")
    log("=============================================")

    if not missing_records:
        log("[INFO] Nie ma brakujących graczy do zlecenia. Kończę po backupie/merge.")
        cleanup_local_missing_lists(args.keep_missing_lists)
        return

    log("")
    log("========== PHASE 2: SETUP + FIDE CONNECTIVITY ==========")
    working_servers = []
    bad_fide = []
    connectivity_rows = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(setup_and_test_server, server, args) for server in backup_ok_servers]

        for future in as_completed(futures):
            ok, server, result, err = future.result()

            connectivity_rows.append({
                "host": server["host"],
                "ok": "yes" if ok else "no",
                "status": result.get("status", "unknown"),
                "home": result.get("home", ""),
                "post": result.get("post", ""),
                "calc": result.get("calc", ""),
                "first_bad": result.get("first_bad", ""),
                "details": result.get("details", ""),
            })

            if ok:
                working_servers.append(server)
            else:
                bad_fide.append((server, err or result.get("details", "")))

    working_servers.sort(key=lambda s: s["index"])

    if not working_servers:
        log("")
        log("[ERROR] Żadna maszyna nie przeszła testu FIDE, więc nic nie startuję.")
        log("[INFO] Szczegóły: fide_connectivity.csv")
        sys.exit(1)

    plan = build_missing_shard_plan(working_servers, missing_records)
    all_missing_path = write_missing_lists(header_lines, missing_records, plan)
    save_plan(plan)

    log("")
    log("========== MISSING SHARD PLAN ==========")
    log(f"missing players:       {len(missing_records)}")
    log(f"working machines:      {len(working_servers)}")
    log(f"scraper workers:       {args.workers}")
    log(f"remote cleanup:        {args.remote_cleanup}")
    log(f"backup mode:           new-only")
    log(f"FIDE test skipped:     {args.skip_fide_test}")
    log(f"clean remote tar.gz:   {CLEAN_REMOTE_TAR_GZ}")
    log(f"plan file:             shard_plan.csv")
    log(f"connectivity file:     fide_connectivity.csv")
    log(f"missing all list:      {all_missing_path}")
    log("========================================")

    for job in plan:
        log(
            f"{job['host']}: missing[{job['missing_start']}:{job['missing_end_exclusive']}] "
            f"count={job['max_players']} ids={job['first_player_id']}..{job['last_player_id']}"
        )

    log("")
    log("========== PHASE 3: UPLOAD SMALL LISTS + START ==========")

    ok_count = 0
    failed_jobs = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_start_job, job, args.workers, args.remote_cleanup) for job in plan]

        for future in as_completed(futures):
            ok, host, err = future.result()

            if ok:
                ok_count += 1
            else:
                failed_jobs.append((host, err))

    log("")
    log("========== ALL DONE ==========")
    log(f"started:          {ok_count}")
    log(f"failed start:     {len(failed_jobs)}")
    log(f"bad FIDE/setup:   {len(bad_fide)}")
    log(f"backup failed:    {len(backup_failed)}")
    log(f"plan:             shard_plan.csv")
    log(f"connectivity:     fide_connectivity.csv")
    log(f"missing lists:    {LOCAL_MISSING_LIST_DIR}")
    log(f"local results:    {LOCAL_FINAL_OUTPUT_DIR}")
    log("==============================")

    cleanup_local_missing_lists(args.keep_missing_lists)

    if failed_jobs:
        log("")
        log("[FAILED START]")
        for host, err in failed_jobs:
            log(f"{host}: {err}")

    if bad_fide:
        log("")
        log("[BAD FIDE / NOT STARTED]")
        for server, err in bad_fide:
            log(f"{server['host']}: {err}")

    if backup_failed:
        log("")
        log("[BACKUP/SSH FAILED]")
        for server, err in backup_failed:
            log(f"{server['host']}: {err}")


if __name__ == "__main__":
    main()
