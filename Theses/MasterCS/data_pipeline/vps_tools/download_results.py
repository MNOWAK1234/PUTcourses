import argparse
import csv
import logging
import os
import posixpath
import shlex
import shutil
import sys
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

import paramiko


# Paramiko przy błędach typu "Error reading SSH protocol banner" potrafi
# wypisywać własny traceback z wewnętrznego wątku transportu. Nie chcemy tego
# w normalnym outputcie — wystarczy nasz krótki "[host] ERROR | ...".
logging.getLogger("paramiko").addHandler(logging.NullHandler())
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)


@contextmanager
def suppress_stderr_unless_verbose():
    if VERBOSE:
        yield
        return

    old_stderr_fd = None
    devnull = None

    try:
        devnull = open(os.devnull, "w")
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        if old_stderr_fd is not None:
            try:
                os.dup2(old_stderr_fd, 2)
            finally:
                os.close(old_stderr_fd)

        if devnull is not None:
            devnull.close()


def compact_exception(e) -> str:
    msg = str(e).replace("\r", " ").replace("\n", " ").strip()
    cls = type(e).__name__

    if "Error reading SSH protocol banner" in msg:
        return "SSH banner error / connection reset"
    if "Authentication failed" in msg or cls == "AuthenticationException":
        return "authentication failed"
    if "No existing session" in msg:
        return "no existing SSH session"
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        return f"{cls}: timeout"
    if not msg:
        return cls

    if len(msg) > 180:
        msg = msg[:177] + "..."

    return f"{cls}: {msg}"


# =========================================
# CONFIG
# =========================================
SCRIPT_DIR = Path(__file__).resolve().parent
SERVERS_CSV = "servers.csv"

USER_DEFAULT = "root"
KEY_FALLBACK_USER = "ubuntu"

REMOTE_OUTPUT_DIR_NAME = "fide_standard_games_by_id"

# Tu trafiają finalnie WSZYSTKIE CSV, razem z lokalnymi wynikami.
LOCAL_FINAL_OUTPUT_DIR = Path("fide_standard_games_by_id")

# Folder techniczny na chwilowe archiwa/rozpakowanie.
LOCAL_WORK_DIR = Path("vps_archives_tmp")

# Jeśli None, skrypt bierze pierwszy plik *.pem z folderu skryptu.
DEFAULT_KEY_FILENAME = None

DELETE_REMOTE_ARCHIVE_AFTER_DOWNLOAD = True
DELETE_LOCAL_ARCHIVE_AFTER_EXTRACT = True
DELETE_EXTRACTED_AFTER_COPY = True

# Sprzątaj stare techniczne archiwa .tar.gz z katalogu scrapera na VPS.
# Usuwa tylko archiwa z głównego katalogu scrapera, nie rusza CSV.
CLEAN_REMOTE_TAR_GZ = True


# =========================================
# LOGGING
# =========================================
VERBOSE = False


def log(message: str):
    print(message, flush=True)


def vlog(message: str):
    if VERBOSE:
        print(message, flush=True)


# =========================================
# HELPERS
# =========================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    value = str(value).strip()
    for ch in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", "."]:
        value = value.replace(ch, "_")
    return value


def find_default_key_path() -> str:
    if DEFAULT_KEY_FILENAME:
        candidate = SCRIPT_DIR / DEFAULT_KEY_FILENAME
        return str(candidate) if candidate.exists() else ""

    pem_files = sorted(SCRIPT_DIR.glob("*.pem"))
    if not pem_files:
        return ""

    return str(pem_files[0])


def default_remote_base_dir_for_user(user: str) -> str:
    user = (user or USER_DEFAULT).strip()
    if user == "root":
        return "/root/fide_scraper"
    return f"/home/{user}/fide_scraper"


def remote_base_dir(server: dict) -> str:
    ssh_user = server.get("ssh_user") or server.get("user") or USER_DEFAULT
    return server.get("remote_base_dir") or default_remote_base_dir_for_user(ssh_user)


def remote_output_dir(server: dict) -> str:
    return f"{remote_base_dir(server)}/{REMOTE_OUTPUT_DIR_NAME}"


# =========================================
# FILE CLASSIFICATION / LOCAL STATE
# =========================================
def get_player_id_from_game_csv_name(name: str):
    name = name.lower()

    if not name.endswith(".csv"):
        return None
    if name.endswith(".tmp.csv"):
        return None
    if name == "run_errors.csv" or name.startswith("run_errors__"):
        return None
    if name.endswith("_errors.csv"):
        return None

    base = name[:-len(".csv")]
    return base if base.isdigit() else None


def get_player_id_from_error_csv_name(name: str):
    name = name.lower()

    if not name.endswith("_errors.csv"):
        return None

    base = name[:-len("_errors.csv")]
    return base if base.isdigit() else None


def classify_csv_name(name: str):
    lower = name.lower()

    if lower.endswith(".tmp.csv"):
        return "tmp", None

    if lower == "run_errors.csv" or lower.startswith("run_errors__"):
        return "run_errors", None

    pid = get_player_id_from_game_csv_name(lower)
    if pid is not None:
        return "real", pid

    pid = get_player_id_from_error_csv_name(lower)
    if pid is not None:
        return "error", pid

    return "other", None


def scan_local_state():
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


def decide_remote_files_to_download(remote_names, local_real_ids, local_error_ids):
    """
    Zasada new-only po poprawce:

    - remote pid.csv pobieramy, jeśli lokalnie NIE ma pid.csv
      ALBO lokalnie istnieje pid_errors.csv dla tego gracza.

      Dzięki temu remote pid.csv naprawia lokalny error-only oraz real+error.

    - remote pid_errors.csv pobieramy tylko jeśli lokalnie nie ma ani pid.csv,
      ani pid_errors.csv. Jeśli lokalnie już mamy error, kolejny error nie
      pomaga; retry powinien zrobić scraper.

    - run_errors.csv, *.tmp.csv i inne pliki pomijamy.
    """
    include_names = []
    remote_real_ids = set()
    remote_error_ids = set()

    skipped_existing_real = 0
    skipped_existing_error = 0
    skipped_run_errors = 0
    skipped_tmp = 0
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

        elif kind == "run_errors":
            skipped_run_errors += 1

        elif kind == "tmp":
            skipped_tmp += 1

        else:
            skipped_other += 1

    return {
        "include_names": include_names,
        "remote_real_ids": remote_real_ids,
        "remote_error_ids": remote_error_ids,
        "skipped_existing_real": skipped_existing_real,
        "skipped_existing_error": skipped_existing_error,
        "skipped_run_errors": skipped_run_errors,
        "skipped_tmp": skipped_tmp,
        "skipped_other": skipped_other,
    }


# =========================================
# LOAD SERVERS
# =========================================
def load_servers():
    if not os.path.exists(SERVERS_CSV):
        raise FileNotFoundError(f"Nie ma pliku: {SERVERS_CSV}")

    default_key_path = find_default_key_path()

    servers = []

    with open(SERVERS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV jest pusty albo nie ma nagłówka.")

        required = {"host", "password"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError("CSV musi mieć stary format kolumn: host,password")

        for i, row in enumerate(reader, start=1):
            host = (row.get("host") or "").strip()
            password = (row.get("password") or "").strip()

            if not host:
                continue

            name = (row.get("name") or host).strip()

            servers.append({
                "name": name,
                "host": host,
                "user": USER_DEFAULT,
                "password": password,
                "key_path": default_key_path,
                "ssh_user": None,
                "remote_base_dir": None,
            })

    return servers


# =========================================
# SSH
# =========================================
def connect_ssh(server):
    password_error = None
    key_error = None

    password = server.get("password") or ""
    key_path = server.get("key_path") or ""

    if password:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            vlog(f"[CONNECT password] {USER_DEFAULT}@{server['host']}")

            with suppress_stderr_unless_verbose():
                ssh.connect(
                    hostname=server["host"],
                    username=USER_DEFAULT,
                    password=password,
                    timeout=30,
                    banner_timeout=30,
                    auth_timeout=30,
                    look_for_keys=False,
                    allow_agent=False,
                )

            server["ssh_user"] = USER_DEFAULT
            server["remote_base_dir"] = default_remote_base_dir_for_user(USER_DEFAULT)

            vlog("[CONNECTED password]")
            return ssh

        except Exception as e:
            password_error = compact_exception(e)
            ssh.close()
            vlog(f"[{server['host']}] password auth failed, trying key if available: {password_error}")

    if key_path:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            vlog(f"[CONNECT key] {KEY_FALLBACK_USER}@{server['host']} key={key_path}")

            with suppress_stderr_unless_verbose():
                ssh.connect(
                    hostname=server["host"],
                    username=KEY_FALLBACK_USER,
                    key_filename=key_path,
                    timeout=30,
                    banner_timeout=30,
                    auth_timeout=30,
                    look_for_keys=False,
                    allow_agent=False,
                )

            server["ssh_user"] = KEY_FALLBACK_USER
            server["remote_base_dir"] = default_remote_base_dir_for_user(KEY_FALLBACK_USER)

            vlog("[CONNECTED key]")
            return ssh

        except Exception as e:
            key_error = compact_exception(e)
            ssh.close()

    details = []

    if password:
        details.append(f"password={password_error or 'not tried'}")
    else:
        details.append("password=empty")

    if key_path:
        details.append(f"key={key_error or 'not tried'}")
    else:
        details.append("key=no *.pem found")

    raise RuntimeError("auth failed; " + "; ".join(details))


def run_remote_command(ssh, command, timeout=None, allow_fail=False):
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
        raise RuntimeError(f"Remote command failed with exit code {exit_code}: {short_err}")

    return exit_code, out, err


def upload_file_sftp(ssh, local_path: Path, remote_path: str):
    sftp = ssh.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def download_file_sftp(ssh, remote_path: str, local_path: Path):
    ensure_dir(local_path.parent)

    sftp = ssh.open_sftp()
    try:
        remote_size = sftp.stat(remote_path).st_size

        if VERBOSE:
            print("\n" + "=" * 80)
            print("[DOWNLOAD]")
            print(f"{remote_path} -> {local_path}")
            print(f"size: {remote_size / (1024 * 1024):.2f} MB")
            print("=" * 80)

            def progress(transferred, total):
                percent = (transferred / total * 100) if total else 0
                print(
                    f"\rProgress: {percent:6.2f}%  "
                    f"{transferred / (1024 * 1024):.2f}/"
                    f"{total / (1024 * 1024):.2f} MB",
                    end="",
                )

            sftp.get(remote_path, str(local_path), callback=progress)
            print("\n[DOWNLOAD DONE]")
        else:
            sftp.get(remote_path, str(local_path))

    finally:
        sftp.close()


# =========================================
# REMOTE HELPERS
# =========================================
def cleanup_remote_tar_gz(ssh, server: dict, reason: str = ""):
    if not CLEAN_REMOTE_TAR_GZ:
        return 0

    rdir = remote_base_dir(server)

    code, out, err = run_remote_command(
        ssh,
        (
            f"if [ -d {rdir!r} ]; then "
            f"COUNT=$(find {rdir!r} -maxdepth 1 -type f -name '*.tar.gz' | wc -l); "
            f"find {rdir!r} -maxdepth 1 -type f -name '*.tar.gz' -delete; "
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
        log(f"[{server['host']}] cleaned old remote tar.gz={deleted}" + (f" ({reason})" if reason else ""))

    return deleted


def list_remote_csv_names(ssh, remote_output_dir_path: str):
    command = (
        f"if [ -d {remote_output_dir_path!r} ]; then "
        f"find {remote_output_dir_path!r} -maxdepth 1 -type f -name '*.csv' -printf '%f\\n' 2>/dev/null; "
        f"fi"
    )

    code, out, err = run_remote_command(ssh, command, timeout=180, allow_fail=True)

    names = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            names.append(line)

    return names


def upload_include_list(ssh, include_names, rdir: str, archive_name: str):
    ensure_dir(LOCAL_WORK_DIR)

    include_name = archive_name.replace(".tar.gz", "_include.txt")
    local_include_path = LOCAL_WORK_DIR / include_name
    remote_include_path = posixpath.join(rdir, include_name)

    with open(local_include_path, "w", encoding="utf-8", newline="\n") as f:
        for name in include_names:
            if "/" in name or "\\" in name or name in ("", ".", ".."):
                continue
            f.write(f"{REMOTE_OUTPUT_DIR_NAME}/{name}\n")

    upload_file_sftp(ssh, local_include_path, remote_include_path)

    try:
        local_include_path.unlink()
    except OSError:
        pass

    return remote_include_path


# =========================================
# LOCAL MERGE
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
            tar.extractall(extract_to)


def copy_csv_to_final(src: Path, final_dir: Path) -> str:
    ensure_dir(final_dir)

    name = src.name
    kind, _ = classify_csv_name(name)

    if kind == "tmp":
        return "skipped_tmp"

    if kind == "run_errors":
        return "skipped_run_errors"

    if kind not in ("real", "error"):
        return "skipped_other"

    dst = final_dir / name

    if not dst.exists():
        shutil.copy2(src, dst)
        return "copied_new"

    shutil.copy2(src, dst)
    return "overwritten"


def merge_archive_into_final(archive_path: Path, prefix: str, remote_error_ids: set):
    extract_to = LOCAL_WORK_DIR / f"extracted_new_only_{prefix}"

    if extract_to.exists():
        shutil.rmtree(extract_to)

    vlog(f"[EXTRACT] {archive_path} -> {extract_to}")
    safe_extract_tar(archive_path, extract_to)

    copied_new = 0
    overwritten = 0
    run_errors = 0
    skipped = 0
    tmp_skipped = 0
    stale_errors_deleted = 0

    csv_files = sorted(extract_to.rglob("*.csv"))

    downloaded_success_player_ids = set()

    for csv_path in csv_files:
        status = copy_csv_to_final(csv_path, LOCAL_FINAL_OUTPUT_DIR)

        if status == "skipped_tmp":
            tmp_skipped += 1
        elif status == "copied_new":
            copied_new += 1
        elif status == "overwritten":
            overwritten += 1
        elif status == "skipped_run_errors":
            run_errors += 1
        elif status.startswith("skipped"):
            skipped += 1
        else:
            copied_new += 1

        kind, pid = classify_csv_name(csv_path.name)
        if kind == "real" and pid is not None:
            downloaded_success_player_ids.add(pid)

    # Jeżeli pobraliśmy czysty pid.csv, a na VPS nie ma pid_errors.csv,
    # to lokalny stary pid_errors.csv jest nieaktualny i można go usunąć.
    for pid in downloaded_success_player_ids:
        if pid in remote_error_ids:
            continue

        local_error_path = LOCAL_FINAL_OUTPUT_DIR / f"{pid}_errors.csv"

        if local_error_path.exists():
            local_error_path.unlink()
            stale_errors_deleted += 1

    if DELETE_EXTRACTED_AFTER_COPY:
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


# =========================================
# DOWNLOAD PROCESS
# =========================================
def make_archive_name(server):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = safe_name(server["name"] or server["host"])
    host_safe = safe_name(server["host"])
    return f"new_only_{prefix}_{host_safe}_{timestamp}.tar.gz"


def get_remote_count_and_size(ssh, remote_output_dir_path: str):
    code, out, err = run_remote_command(
        ssh,
        (
            f"CSV_COUNT=$(find {remote_output_dir_path!r} -name '*.csv' 2>/dev/null | wc -l); "
            f"SIZE=$(du -sh {remote_output_dir_path!r} 2>/dev/null | awk '{{print $1}}'); "
            f"echo \"$CSV_COUNT|$SIZE\""
        ),
        timeout=120,
    )

    line = out.strip().splitlines()[-1] if out.strip() else "0|?"
    parts = line.split("|", 1)

    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return "?", "?"


def process_server(server):
    host = server["host"]
    name = server["name"]
    prefix = safe_name(f"{name}_{host}")

    archive_name = make_archive_name(server)
    local_archive_path = LOCAL_WORK_DIR / archive_name

    ssh = None

    try:
        # Aktualizujemy lokalny stan per maszyna, bo poprzednia maszyna mogła już coś domergować.
        local_state = scan_local_state()

        log(f"[{host}] start | local real={len(local_state['real_ids'])} error={len(local_state['error_ids'])}")

        ssh = connect_ssh(server)

        rdir = remote_base_dir(server)
        remote_output_dir_path = remote_output_dir(server)
        remote_archive_path = posixpath.join(rdir, archive_name)

        cleanup_remote_tar_gz(ssh, server, reason="before backup")

        code, _, _ = run_remote_command(
            ssh,
            f"test -d {remote_output_dir_path!r}",
            timeout=60,
            allow_fail=True,
        )

        if code != 0:
            raise RuntimeError(f"remote output dir missing: {remote_output_dir_path}")

        csv_count, remote_size = get_remote_count_and_size(ssh, remote_output_dir_path)
        log(f"[{host}] ssh_user={server.get('ssh_user')} remote={rdir}")
        log(f"[{host}] remote csv={csv_count}, size={remote_size}")

        remote_names = list_remote_csv_names(ssh, remote_output_dir_path)
        decision = decide_remote_files_to_download(
            remote_names=remote_names,
            local_real_ids=local_state["real_ids"],
            local_error_ids=local_state["error_ids"],
        )
        include_names = decision["include_names"]

        log(
            f"[{host}] new_to_download={len(include_names)} "
            f"skip_real={decision['skipped_existing_real']} "
            f"skip_error={decision['skipped_existing_error']} "
            f"skip_tmp={decision['skipped_tmp']} "
            f"skip_run_errors={decision['skipped_run_errors']}"
        )

        if not include_names:
            cleanup_remote_tar_gz(ssh, server, reason="nothing new")
            log(f"[{host}] OK | nothing new")
            return True, {
                "remote_csv": len(remote_names),
                "new_to_download": 0,
                "csv_found": 0,
                "copied_new": 0,
                "overwritten": 0,
                "run_errors": 0,
                "skipped": 0,
                "tmp_skipped": 0,
                "stale_errors_deleted": 0,
            }

        remote_include_path = upload_include_list(ssh, include_names, rdir, archive_name)

        tar_command = (
            f"cd {rdir!r} && "
            f"tar --warning=no-file-changed "
            f"--ignore-failed-read "
            f"-czf {archive_name!r} "
            f"-T {remote_include_path!r}"
        )

        log(f"[{host}] packing new-only.")
        run_remote_command(ssh, tar_command, timeout=None)

        log(f"[{host}] downloading.")
        download_file_sftp(ssh, remote_archive_path, local_archive_path)

        if DELETE_REMOTE_ARCHIVE_AFTER_DOWNLOAD:
            run_remote_command(
                ssh,
                f"rm -f {remote_archive_path!r} {remote_include_path!r}",
                timeout=60,
                allow_fail=True,
            )

        cleanup_remote_tar_gz(ssh, server, reason="after backup")

        log(f"[{host}] merging.")
        stats = merge_archive_into_final(
            archive_path=local_archive_path,
            prefix=prefix,
            remote_error_ids=decision["remote_error_ids"],
        )

        stats["remote_csv"] = len(remote_names)
        stats["new_to_download"] = len(include_names)

        if DELETE_LOCAL_ARCHIVE_AFTER_EXTRACT:
            local_archive_path.unlink(missing_ok=True)

        log(
            f"[{host}] OK | remote_csv={stats['remote_csv']} "
            f"new_to_download={stats['new_to_download']} "
            f"found={stats['csv_found']} "
            f"new={stats['copied_new']} overwritten={stats['overwritten']} "
            f"stale_errors_deleted={stats['stale_errors_deleted']} "
            f"skipped={stats['skipped']}"
        )

        return True, stats

    except Exception as e:
        log(f"[{host}] ERROR | {compact_exception(e)}")
        return False, None

    finally:
        if ssh is not None:
            ssh.close()
            vlog("[SSH CLOSED]")


# =========================================
# CLI
# =========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pobierz tylko nowe/użyteczne wyniki z VPS i scal do lokalnego fide_standard_games_by_id.")
    parser.add_argument("--verbose", action="store_true", help="Pokaż pełne komendy, stdout/stderr i progress pobierania.")
    parser.add_argument("--no-clean-remote-tar-gz", action="store_true", help="Nie usuwaj starych *.tar.gz z katalogu scrapera na VPS.")
    parser.add_argument("--servers", default=SERVERS_CSV, help="CSV z host,password. Domyślnie: servers.csv")
    return parser.parse_args()


def main():
    global VERBOSE, SERVERS_CSV, CLEAN_REMOTE_TAR_GZ

    args = parse_args()
    VERBOSE = args.verbose
    SERVERS_CSV = args.servers
    CLEAN_REMOTE_TAR_GZ = not args.no_clean_remote_tar_gz

    ensure_dir(LOCAL_FINAL_OUTPUT_DIR)
    ensure_dir(LOCAL_WORK_DIR)

    servers = load_servers()

    if not servers:
        print("[ERROR] Brak serwerów w CSV.")
        sys.exit(1)

    local_state = scan_local_state()
    local_real = len(local_state["real_ids"])
    local_error = len(local_state["error_ids"])
    local_any = len(local_state["real_ids"] | local_state["error_ids"])

    log(f"[INFO] servers={len(servers)}")
    log(f"[INFO] final={LOCAL_FINAL_OUTPUT_DIR}")
    log(f"[INFO] local state: real={local_real} error={local_error} any={local_any}")
    log("[INFO] policy: download only missing/useful files")
    log("[INFO] policy: remote pid.csv downloads if local pid.csv is missing")
    log("[INFO] policy: remote pid_errors.csv downloads only if local has neither pid.csv nor pid_errors.csv")
    log("[INFO] run_errors.csv policy: skip / do not save")
    log("[INFO] stale player error policy: delete local *_errors.csv if downloaded VPS has clean player CSV")
    log(f"[INFO] clean remote tar.gz: {CLEAN_REMOTE_TAR_GZ}")

    ok = 0
    total_remote_csv = 0
    total_new_to_download = 0
    total_found = 0
    total_new = 0
    total_overwritten = 0
    total_run_errors = 0
    total_skipped = 0
    total_stale_errors_deleted = 0

    for server in servers:
        success, stats = process_server(server)

        if success:
            ok += 1
            total_remote_csv += stats.get("remote_csv", 0)
            total_new_to_download += stats.get("new_to_download", 0)
            total_found += stats.get("csv_found", 0)
            total_new += stats.get("copied_new", 0)
            total_overwritten += stats.get("overwritten", 0)
            total_run_errors += stats.get("run_errors", 0)
            total_skipped += stats.get("skipped", 0)
            total_stale_errors_deleted += stats.get("stale_errors_deleted", 0)

    log("")
    log("========== ALL DONE ==========")
    log(f"OK:                    {ok}")
    log(f"FAILED:                {len(servers) - ok}")
    log(f"REMOTE_CSV_SEEN:       {total_remote_csv}")
    log(f"NEW_TO_DOWNLOAD:       {total_new_to_download}")
    log(f"ARCHIVE_CSV_FOUND:     {total_found}")
    log(f"NEW_LOCAL_FILES:       {total_new}")
    log(f"OVERWRITTEN:           {total_overwritten}")
    log(f"STALE_ERRORS_DELETED:  {total_stale_errors_deleted}")
    log(f"RUN_ERRORS_SKIPPED:    {total_run_errors}")
    log(f"SKIPPED:               {total_skipped}")
    log(f"FINAL:                 {LOCAL_FINAL_OUTPUT_DIR}")
    log("==============================")


if __name__ == "__main__":
    main()
