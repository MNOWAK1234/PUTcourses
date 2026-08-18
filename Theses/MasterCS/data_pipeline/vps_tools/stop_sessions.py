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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko


DEFAULT_SERVERS_CSV = "servers.csv"

DEFAULT_ROOT_USER = "root"
DEFAULT_KEY_USER = "ubuntu"

TMUX_SESSION = "fide"
REMOTE_OUTPUT_DIR_NAME = "fide_standard_games_by_id"

LOCAL_FINAL_OUTPUT_DIR = Path("fide_standard_games_by_id")
LOCAL_WORK_DIR = Path("vps_archives_tmp")

DEFAULT_KEY_FILENAME = None

DELETE_REMOTE_ARCHIVE_AFTER_DOWNLOAD = True
DELETE_LOCAL_ARCHIVE_AFTER_EXTRACT = True
DELETE_EXTRACTED_AFTER_COPY = True
CLEAN_REMOTE_TAR_GZ = True

SSH_CONNECT_TIMEOUT = 45
SSH_BANNER_TIMEOUT = 90
SSH_AUTH_TIMEOUT = 45
SSH_CONNECT_RETRIES = 3
SSH_RETRY_SLEEP_SECONDS = 8

VERBOSE = False


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


def log(msg: str):
    print(msg, flush=True)


def vlog(msg: str):
    if VERBOSE:
        print(msg, flush=True)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    value = str(value).strip()
    for ch in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", "."]:
        value = value.replace(ch, "_")
    return value


def short_error(e: Exception, limit: int = 180) -> str:
    text = f"{type(e).__name__}: {str(e)}".replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit - 3] + "..."
    return text


def find_default_key_path() -> str:
    base = script_dir()

    if DEFAULT_KEY_FILENAME:
        candidate = base / DEFAULT_KEY_FILENAME
        return str(candidate) if candidate.exists() else ""

    pem_files = sorted(base.glob("*.pem"))
    if not pem_files:
        return ""

    return str(pem_files[0])


def remote_base_dir_for_user(user: str) -> str:
    if user == "root":
        return "/root/fide_scraper"
    return f"/home/{user}/fide_scraper"


def remote_output_dir_for_user(user: str) -> str:
    return f"{remote_base_dir_for_user(user)}/{REMOTE_OUTPUT_DIR_NAME}"


def get_player_id_from_game_csv_name(name: str):
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

            # Download real CSV if locally there is no real CSV yet
            # OR if local state still has an error for this player.
            # This fixes both local error-only and local real+error state.
            if pid not in local_real_ids or pid in local_error_ids:
                include_names.append(name)
            else:
                skipped_existing_real += 1

        elif kind == "error":
            remote_error_ids.add(pid)

            # Download error CSV only for completely unseen players.
            # If local already has real or error, skip.
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


def load_servers(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie ma pliku: {path}")

    servers = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV jest pusty albo nie ma nagłówka.")

        required = {"host", "password"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError("CSV musi mieć kolumny: host,password")

        for i, row in enumerate(reader, start=1):
            host = (row.get("host") or "").strip()
            password = (row.get("password") or "").strip()

            if not host:
                continue

            servers.append({
                "index": i,
                "name": (row.get("name") or f"vps_{i}").strip(),
                "host": host,
                "password": password,
            })

    return servers


def new_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return ssh


def connect_once(host: str, user: str, *, password: str = "", key_path: str = ""):
    ssh = new_ssh_client()

    kwargs = {
        "hostname": host,
        "username": user,
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

    try:
        ssh.connect(**kwargs)
        return ssh
    except Exception:
        try:
            ssh.close()
        except Exception:
            pass
        raise


def connect_with_retries(host: str, user: str, label: str, *, password: str = "", key_path: str = ""):
    errors = []

    for attempt in range(1, SSH_CONNECT_RETRIES + 1):
        try:
            ssh = connect_once(host, user, password=password, key_path=key_path)

            if attempt > 1:
                vlog(f"[{host}] SSH {label} connected after retry {attempt}/{SSH_CONNECT_RETRIES}")

            return ssh

        except Exception as e:
            msg = short_error(e)
            errors.append(f"{attempt}/{SSH_CONNECT_RETRIES}: {msg}")

            if attempt < SSH_CONNECT_RETRIES:
                vlog(f"[{host}] SSH {label} failed {attempt}/{SSH_CONNECT_RETRIES}: {msg}; retry in {SSH_RETRY_SLEEP_SECONDS}s")
                time.sleep(SSH_RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"{label} failed after {SSH_CONNECT_RETRIES} tries: " + " | ".join(errors[-3:]))


def connect_ssh(server: dict, key_path: str):
    host = server["host"]
    password = server.get("password") or ""
    errors = []

    if password:
        try:
            ssh = connect_with_retries(
                host,
                DEFAULT_ROOT_USER,
                "root/password",
                password=password,
            )
            return ssh, DEFAULT_ROOT_USER, "password"
        except Exception as e:
            errors.append(f"password={short_error(e)}")
            vlog(f"[{host}] password auth failed: {short_error(e)}")

    if key_path:
        try:
            ssh = connect_with_retries(
                host,
                DEFAULT_KEY_USER,
                "ubuntu/key",
                key_path=key_path,
            )
            return ssh, DEFAULT_KEY_USER, "pem"
        except Exception as e:
            errors.append(f"pem={short_error(e)}")
            vlog(f"[{host}] pem auth failed: {short_error(e)}")
    else:
        errors.append("pem=brak pliku *.pem w folderze skryptu")

    raise RuntimeError("; ".join(errors))


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
    kind, pid = classify_csv_name(name)

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
    extract_to = LOCAL_WORK_DIR / f"extracted_stop_new_only_{prefix}"

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

    downloaded_success_player_ids = set()

    for csv_path in csv_files:
        status = copy_csv_to_final(csv_path, LOCAL_FINAL_OUTPUT_DIR)

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

        kind, pid = classify_csv_name(csv_path.name)
        if kind == "real" and pid is not None:
            downloaded_success_player_ids.add(pid)

    # If we downloaded a clean pid.csv and the remote did not have pid_errors.csv,
    # remove stale local pid_errors.csv.
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


def stop_scraper_and_tmux(ssh):
    command = (
        f"tmux send-keys -t {shlex.quote(TMUX_SESSION)} C-c 2>/dev/null || true; "
        f"sleep 5; "
        f"tmux kill-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null || true; "
        f"pkill -INT -f scraperFaster.py 2>/dev/null || true; "
        f"sleep 2; "
        f"pkill -KILL -f scraperFaster.py 2>/dev/null || true; "
        f"echo stopped"
    )

    run_command(ssh, command, timeout=90, allow_fail=True)


def get_remote_status(ssh, user: str):
    rdir = remote_base_dir_for_user(user)
    out_dir = remote_output_dir_for_user(user)

    command = f"""
if command -v tmux >/dev/null 2>&1 && tmux has-session -t {shlex.quote(TMUX_SESSION)} >/dev/null 2>&1; then
  echo tmux=yes
else
  echo tmux=no
fi

if ps -eo pid,ppid,stat,cmd | grep '[s]craperFaster.py' >/dev/null 2>&1; then
  echo scraper=yes
else
  echo scraper=no
fi

if test -d {shlex.quote(out_dir)}; then
  echo csv=$(find {shlex.quote(out_dir)} -maxdepth 1 -type f -name '*.csv' 2>/dev/null | wc -l)
else
  echo csv=0
fi

if test -f {shlex.quote(rdir)}/run.log; then
  echo runlog_size=$(wc -c < {shlex.quote(rdir)}/run.log)
else
  echo runlog_size=0
fi
""".strip()

    code, out, err = run_command(ssh, command, timeout=60, allow_fail=True)

    status = {
        "tmux": "unknown",
        "scraper": "unknown",
        "csv": "0",
        "runlog_size": "0",
    }

    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k in status:
            status[k] = v

    return status


def remote_dir_exists(ssh, path: str) -> bool:
    code, out, err = run_command(
        ssh,
        f"test -d {shlex.quote(path)} && echo yes || echo no",
        timeout=30,
        allow_fail=True,
    )

    return out.strip().splitlines()[-1] == "yes"


def cleanup_remote_tar_gz(ssh, user: str, host: str, reason: str = ""):
    if not CLEAN_REMOTE_TAR_GZ:
        return 0

    rdir = remote_base_dir_for_user(user)

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
        log(f"[{host}] cleaned old remote tar.gz={deleted}" + (f" ({reason})" if reason else ""))

    return deleted


def list_remote_csv_names(ssh, out_dir: str):
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


def upload_include_list(ssh, include_names, rdir: str, archive_name: str):
    ensure_dir(LOCAL_WORK_DIR)

    include_name = archive_name.replace(".tar.gz", "_include.txt")
    local_include_path = LOCAL_WORK_DIR / include_name
    remote_include_path = posixpath.join(rdir, include_name)

    with open(local_include_path, "w", encoding="utf-8", newline="\n") as f:
        for name in include_names:
            # Basename only, no paths.
            if "/" in name or "\\" in name or name in ("", ".", ".."):
                continue
            f.write(f"{REMOTE_OUTPUT_DIR_NAME}/{name}\n")

    upload_file(ssh, local_include_path, remote_include_path)

    try:
        local_include_path.unlink()
    except OSError:
        pass

    return remote_include_path


def backup_remote_results_new_only(ssh, server: dict, user: str, local_state: dict):
    host = server["host"]
    prefix = safe_name(f"{server.get('name') or host}_{host}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    rdir = remote_base_dir_for_user(user)
    out_dir = remote_output_dir_for_user(user)

    if not remote_dir_exists(ssh, rdir):
        return None

    if not remote_dir_exists(ssh, out_dir):
        return None

    cleanup_remote_tar_gz(ssh, user, host, reason="before backup")

    remote_names = list_remote_csv_names(ssh, out_dir)

    decision = decide_remote_files_to_download(
        remote_names=remote_names,
        local_real_ids=local_state["real_ids"],
        local_error_ids=local_state["error_ids"],
    )

    include_names = decision["include_names"]

    log(
        f"[{host}] remote csv={len(remote_names)} new_to_download={len(include_names)} "
        f"skip_real={decision['skipped_existing_real']} "
        f"skip_error={decision['skipped_existing_error']}"
    )

    if not include_names:
        cleanup_remote_tar_gz(ssh, user, host, reason="nothing new")
        return {
            "csv_found": 0,
            "copied_new": 0,
            "overwritten": 0,
            "run_errors": 0,
            "skipped": 0,
            "tmp_skipped": 0,
            "stale_errors_deleted": 0,
            "remote_csv": len(remote_names),
            "new_to_download": 0,
        }

    archive_name = f"stop_new_only_{prefix}_{timestamp}.tar.gz"
    remote_archive_path = posixpath.join(rdir, archive_name)
    local_archive_path = LOCAL_WORK_DIR / archive_name

    remote_include_path = upload_include_list(ssh, include_names, rdir, archive_name)

    tar_command = (
        f"cd {shlex.quote(rdir)} && "
        f"tar --warning=no-file-changed "
        f"--ignore-failed-read "
        f"-czf {shlex.quote(archive_name)} "
        f"-T {shlex.quote(remote_include_path)}"
    )

    log(f"[{host}] packing new-only files={len(include_names)}...")
    run_command(ssh, tar_command, timeout=None)

    log(f"[{host}] downloading archive...")
    download_file(ssh, remote_archive_path, local_archive_path)

    if DELETE_REMOTE_ARCHIVE_AFTER_DOWNLOAD:
        run_command(
            ssh,
            f"rm -f {shlex.quote(remote_archive_path)} {shlex.quote(remote_include_path)}",
            timeout=60,
            allow_fail=True,
        )

    cleanup_remote_tar_gz(ssh, user, host, reason="after backup")

    log(f"[{host}] merging...")
    stats = merge_archive_into_final(
        archive_path=local_archive_path,
        prefix=prefix,
        remote_error_ids=decision["remote_error_ids"],
    )

    stats["remote_csv"] = len(remote_names)
    stats["new_to_download"] = len(include_names)

    if DELETE_LOCAL_ARCHIVE_AFTER_EXTRACT:
        try:
            local_archive_path.unlink()
        except OSError:
            pass

    return stats


def delete_remote_output_if_requested(ssh, user: str):
    out_dir = remote_output_dir_for_user(user)

    run_command(
        ssh,
        (
            f"rm -rf {shlex.quote(out_dir)} && "
            f"mkdir -p {shlex.quote(out_dir)} && "
            f"CSV_LEFT=$(find {shlex.quote(out_dir)} -name '*.csv' 2>/dev/null | wc -l); "
            f"echo CSV_LEFT=$CSV_LEFT"
        ),
        timeout=180,
    )


def process_server(server: dict, key_path: str, local_state: dict, args):
    host = server["host"]
    ssh = None

    try:
        log(f"[{host}] connecting...")
        ssh, user, auth = connect_ssh(server, key_path)

        before = get_remote_status(ssh, user)
        log(
            f"[{host}] auth={user}/{auth} before: "
            f"tmux={before['tmux']} scraper={before['scraper']} csv={before['csv']}"
        )

        if not args.no_stop:
            log(f"[{host}] stopping tmux/process...")
            stop_scraper_and_tmux(ssh)

            after_stop = get_remote_status(ssh, user)
            log(
                f"[{host}] stopped: "
                f"tmux={after_stop['tmux']} scraper={after_stop['scraper']} csv={after_stop['csv']}"
            )
        else:
            log(f"[{host}] stop skipped by --no-stop")

        stats = None

        if not args.no_backup:
            log(f"[{host}] backup new-only...")
            stats = backup_remote_results_new_only(ssh, server, user, local_state)

            if stats is None:
                log(f"[{host}] backup skipped: no remote output")
            else:
                log(
                    f"[{host}] backup merged | remote_csv={stats.get('remote_csv', 0)} "
                    f"new_to_download={stats.get('new_to_download', 0)} "
                    f"found={stats['csv_found']} "
                    f"new={stats['copied_new']} overwritten={stats['overwritten']} "
                    f"stale_errors_deleted={stats['stale_errors_deleted']} "
                    f"skipped={stats['skipped']}"
                )
        else:
            log(f"[{host}] backup skipped by --no-backup")

        if args.delete_remote_output:
            log(f"[{host}] deleting remote output...")
            delete_remote_output_if_requested(ssh, user)
            log(f"[{host}] remote output deleted")
        else:
            log(f"[{host}] remote output NOT deleted")

        final = get_remote_status(ssh, user)

        return True, host, (
            f"OK auth={user}/{auth} "
            f"final_tmux={final['tmux']} final_scraper={final['scraper']} "
            f"csv={final['csv']}"
        ), stats

    except Exception as e:
        return False, host, short_error(e, 300), None

    finally:
        if ssh is not None:
            try:
                ssh.close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stop all tmux/scraper sessions, then download only NEW/USEFUL CSV files based on local fide_standard_games_by_id."
    )

    parser.add_argument("--servers", default=DEFAULT_SERVERS_CSV, help="CSV host,password. Domyślnie servers.csv")
    parser.add_argument("--parallel", type=int, default=5, help="Ile maszyn równolegle.")
    parser.add_argument("--verbose", action="store_true", help="Pełne komendy i logi.")
    parser.add_argument("--no-backup", action="store_true", help="Nie rób backupu. Domyślnie backup jest robiony.")
    parser.add_argument("--no-stop", action="store_true", help="Nie zatrzymuj tmux/procesu przed backupem.")
    parser.add_argument("--no-clean-remote-tar-gz", action="store_true", help="Nie sprzątaj starych *.tar.gz z katalogu scrapera.")
    parser.add_argument(
        "--delete-remote-output",
        action="store_true",
        default=False,
        help="OPCJONALNE: usuń zdalny folder fide_standard_games_by_id po backupie. Domyślnie false, nic nie usuwa.",
    )
    parser.add_argument("--connect-retries", type=int, default=SSH_CONNECT_RETRIES)
    parser.add_argument("--retry-sleep", type=int, default=SSH_RETRY_SLEEP_SECONDS)
    parser.add_argument("--ssh-timeout", type=int, default=SSH_CONNECT_TIMEOUT)
    parser.add_argument("--banner-timeout", type=int, default=SSH_BANNER_TIMEOUT)
    parser.add_argument("--auth-timeout", type=int, default=SSH_AUTH_TIMEOUT)

    return parser.parse_args()


def main():
    global VERBOSE, CLEAN_REMOTE_TAR_GZ
    global SSH_CONNECT_RETRIES, SSH_RETRY_SLEEP_SECONDS, SSH_CONNECT_TIMEOUT, SSH_BANNER_TIMEOUT, SSH_AUTH_TIMEOUT

    args = parse_args()
    VERBOSE = args.verbose
    CLEAN_REMOTE_TAR_GZ = not args.no_clean_remote_tar_gz

    SSH_CONNECT_RETRIES = max(1, args.connect_retries)
    SSH_RETRY_SLEEP_SECONDS = max(0, args.retry_sleep)
    SSH_CONNECT_TIMEOUT = max(5, args.ssh_timeout)
    SSH_BANNER_TIMEOUT = max(5, args.banner_timeout)
    SSH_AUTH_TIMEOUT = max(5, args.auth_timeout)

    ensure_dir(LOCAL_FINAL_OUTPUT_DIR)
    ensure_dir(LOCAL_WORK_DIR)

    servers = load_servers(args.servers)
    key_path = find_default_key_path()

    local_state = scan_local_state()
    local_real = len(local_state["real_ids"])
    local_error = len(local_state["error_ids"])
    local_any = len(local_state["real_ids"] | local_state["error_ids"])

    log(f"[INFO] servers={len(servers)}")
    log(f"[INFO] local results={LOCAL_FINAL_OUTPUT_DIR}")
    log(f"[INFO] work dir={LOCAL_WORK_DIR}")
    log(f"[INFO] backup={'no' if args.no_backup else 'yes'}")
    log(f"[INFO] stop={'no' if args.no_stop else 'yes'}")
    log(f"[INFO] delete remote output={args.delete_remote_output}")
    log(f"[INFO] clean remote tar.gz={CLEAN_REMOTE_TAR_GZ}")
    log(f"[INFO] pem={key_path if key_path else 'not found'}")
    log(f"[INFO] local state: real={local_real} error={local_error} any={local_any}")
    log("[INFO] policy: download only remote files missing/useful locally")
    log("[INFO] policy: remote pid.csv downloads if local pid.csv is missing")
    log("[INFO] policy: remote pid_errors.csv downloads only if local has neither pid.csv nor pid_errors.csv")
    log("")

    ok_count = 0
    failed = []

    total_remote_csv = 0
    total_new_to_download = 0
    total_found = 0
    total_new = 0
    total_overwritten = 0
    total_stale_errors_deleted = 0
    total_skipped = 0

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = [
            executor.submit(process_server, server, key_path, local_state, args)
            for server in servers
        ]

        for future in as_completed(futures):
            ok, host, msg, stats = future.result()

            if ok:
                ok_count += 1
                log(f"[{host}] DONE | {msg}")

                if stats:
                    total_remote_csv += stats.get("remote_csv", 0)
                    total_new_to_download += stats.get("new_to_download", 0)
                    total_found += stats.get("csv_found", 0)
                    total_new += stats.get("copied_new", 0)
                    total_overwritten += stats.get("overwritten", 0)
                    total_stale_errors_deleted += stats.get("stale_errors_deleted", 0)
                    total_skipped += stats.get("skipped", 0)
            else:
                failed.append((host, msg))
                log(f"[{host}] ERROR | {msg}")

    log("")
    log("========== ALL DONE ==========")
    log(f"OK:                    {ok_count}")
    log(f"FAILED:                {len(failed)}")
    log(f"REMOTE_CSV_SEEN:       {total_remote_csv}")
    log(f"NEW_TO_DOWNLOAD:       {total_new_to_download}")
    log(f"ARCHIVE_CSV_FOUND:     {total_found}")
    log(f"NEW_LOCAL_FILES:       {total_new}")
    log(f"OVERWRITTEN:           {total_overwritten}")
    log(f"STALE_ERRORS_DELETED:  {total_stale_errors_deleted}")
    log(f"SKIPPED:               {total_skipped}")
    log(f"RESULTS:               {LOCAL_FINAL_OUTPUT_DIR}")
    log("==============================")

    if failed:
        log("")
        log("[FAILED]")
        for host, msg in failed:
            log(f"{host}: {msg}")


if __name__ == "__main__":
    main()
