import os
import re
import argparse
import csv
import time
import html as html_lib
import signal
import threading
from typing import Optional, List, Dict
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from tqdm import tqdm


# =========================================
# CONFIG
# =========================================

BASE_URL = "https://ratings.fide.com/"
PROFILE_CALCULATIONS_URL = BASE_URL + "profile/{player_id}/calculations"
MORE_PERIODS_URL = BASE_URL + "a_calculations.phtml"
CALC_AJAX_URL = BASE_URL + "a_indv_calculations.php"

# Plik pobrany z oficjalnej strony FIDE:
# https://ratings.fide.com/download_lists.phtml
STANDARD_RATING_LIST_PATH = r"standard_rating_list.txt"

OUTPUT_DIR = "fide_standard_games_by_id"

# Domyślnie: cały plik od początku. Można nadpisać argumentami CLI.
MAX_PLAYERS = None
START_OFFSET = 0

PLAYER_WORKERS = 8
BATCH_SIZE = PLAYER_WORKERS * 2

RATING_TYPE = 0
RATING_TYPE_NAME = "standard"

REQUEST_TIMEOUT = 12
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 0.75
REQUEST_SLEEP_BETWEEN_CALLS = 0.0

# Najważniejsza zmiana po eksperymencie:
# True = tylko POST /a_calculations.phtml, bez GET /profile/{id}/calculations
POST_ONLY_PERIOD_INDEX = True

SKIP_DONE_IF_CSV_EXISTS_AND_NO_ERRORS = True
OVERWRITE_PLAYER_ON_RETRY = True


# =========================================
# CLI ARGUMENTS
# =========================================

def parse_optional_int(value: Optional[str]) -> Optional[int]:
    """
    Pozwala podać liczbę albo None/null/all, np.:
      --max-players 5000
      --max-players none
    """
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in ("", "none", "null", "all"):
        return None

    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scraper FIDE calculations with configurable shard parameters."
    )

    parser.add_argument(
        "--rating-list",
        "--standard-rating-list",
        dest="rating_list",
        default=STANDARD_RATING_LIST_PATH,
        help=f"Path to standard_rating_list.txt. Default: {STANDARD_RATING_LIST_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=START_OFFSET,
        help=f"How many players to skip from the rating list. Default: {START_OFFSET}",
    )
    parser.add_argument(
        "--max-players",
        type=parse_optional_int,
        default=MAX_PLAYERS,
        help="How many players to scrape. Use none/all for the rest of the file. Default: none",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PLAYER_WORKERS,
        help=f"Number of concurrent player workers. Default: {PLAYER_WORKERS}",
    )
    parser.add_argument(
        "--batch-size",
        type=parse_optional_int,
        default=None,
        help="Max queued futures. Default: workers * 2",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help=f"HTTP request timeout in seconds. Default: {REQUEST_TIMEOUT}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=REQUEST_RETRIES,
        help=f"HTTP retries per request. Default: {REQUEST_RETRIES}",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=REQUEST_BACKOFF,
        help=f"Retry backoff base in seconds. Default: {REQUEST_BACKOFF}",
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=REQUEST_SLEEP_BETWEEN_CALLS,
        help=f"Sleep after every successful HTTP call. Default: {REQUEST_SLEEP_BETWEEN_CALLS}",
    )

    return parser.parse_args()


def apply_cli_config(args):
    global STANDARD_RATING_LIST_PATH
    global OUTPUT_DIR
    global START_OFFSET
    global MAX_PLAYERS
    global PLAYER_WORKERS
    global BATCH_SIZE
    global REQUEST_TIMEOUT
    global REQUEST_RETRIES
    global REQUEST_BACKOFF
    global REQUEST_SLEEP_BETWEEN_CALLS

    if args.start_offset < 0:
        raise ValueError("--start-offset must be >= 0")

    if args.max_players is not None and args.max_players < 0:
        raise ValueError("--max-players must be >= 0 or none/all")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1 or none")

    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1")

    if args.retries < 1:
        raise ValueError("--retries must be >= 1")

    if args.backoff < 0:
        raise ValueError("--backoff must be >= 0")

    if args.sleep_between_calls < 0:
        raise ValueError("--sleep-between-calls must be >= 0")

    STANDARD_RATING_LIST_PATH = args.rating_list
    OUTPUT_DIR = args.output_dir
    START_OFFSET = args.start_offset
    MAX_PLAYERS = args.max_players
    PLAYER_WORKERS = args.workers
    BATCH_SIZE = args.batch_size if args.batch_size is not None else PLAYER_WORKERS * 2
    REQUEST_TIMEOUT = args.timeout
    REQUEST_RETRIES = args.retries
    REQUEST_BACKOFF = args.backoff
    REQUEST_SLEEP_BETWEEN_CALLS = args.sleep_between_calls


# =========================================
# STOP FLAG
# =========================================

STOP_REQUESTED = False


def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[STOP] Ctrl+C detected. Stopping after current active requests...")


signal.signal(signal.SIGINT, handle_sigint)


def should_stop() -> bool:
    return STOP_REQUESTED


# =========================================
# PARSER
# =========================================

try:
    import lxml  # noqa: F401
    BS4_PARSER = "lxml"
except Exception:
    BS4_PARSER = "html.parser"


# =========================================
# CSV COLUMNS
# =========================================

GAMES_COLUMNS = [
    "player_id",
    "player_name",
    "fed",
    "sex",
    "standard_rating_from_list",
    "standard_games_from_list",
    "birth_year",
    "period",
    "rating_type",
    "rating_type_name",
    "event_id",
    "event_name",
    "date_from",
    "date_to",
    "player_rating",
    "event_rc",
    "event_w",
    "event_n",
    "event_chg",
    "event_k",
    "event_k_chg",
    "opponent_fide_id",
    "opponent_name",
    "opponent_rating",
    "display_opponent_rating",
    "score",
    "color",
    "star_400_rule",
]

ERROR_COLUMNS = [
    "logged_at",
    "player_id",
    "player_name",
    "period",
    "stage",
    "error",
]

RUN_ERROR_COLUMNS = [
    "logged_at",
    "player_id",
    "player_name",
    "stage",
    "error",
]


# =========================================
# REGEX
# =========================================

SPACE_RE = re.compile(r"\s+")
NON_NUM_RE = re.compile(r"[^\d\.\-]")
EVENT_ID_RE = re.compile(r"event=(\d+)")
HREF_RE = re.compile(
    r'href\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|([^>\s]+))',
    re.IGNORECASE,
)

SUSPICIOUS_TERMS = (
    "captcha",
    "cloudflare",
    "access denied",
    "forbidden",
    "too many requests",
    "<!doctype",
)


# =========================================
# HELPERS
# =========================================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ").replace("&nbsp;", " ")
    return SPACE_RE.sub(" ", text).strip()


def parse_int(value):
    value = clean_text(value)
    if value == "":
        return None

    value = value.replace("*", "").replace(",", ".")
    value = NON_NUM_RE.sub("", value)

    if value in ("", "-", "."):
        return None

    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def parse_float(value):
    value = clean_text(value)
    if value == "":
        return None

    value = value.replace(",", ".")
    value = NON_NUM_RE.sub("", value)

    if value in ("", "-", "."):
        return None

    try:
        return float(value)
    except ValueError:
        return None


def decode_line(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def output_games_path(player_id: int) -> str:
    return os.path.join(OUTPUT_DIR, f"{player_id}.csv")


def output_tmp_games_path(player_id: int) -> str:
    return os.path.join(OUTPUT_DIR, f"{player_id}.tmp.csv")


def output_errors_path(player_id: int) -> str:
    return os.path.join(OUTPUT_DIR, f"{player_id}_errors.csv")


def output_tmp_errors_path(player_id: int) -> str:
    return os.path.join(OUTPUT_DIR, f"{player_id}_errors.tmp.csv")


def run_errors_path() -> str:
    return os.path.join(OUTPUT_DIR, "run_errors.csv")


def csv_has_data_rows(path: str) -> bool:
    if not os.path.exists(path):
        return False

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            return len(rows) > 1
    except Exception:
        return False


def init_csv_if_needed(path: str, columns: List[str]):
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def write_csv_atomic(path: str, tmp_path: str, columns: List[str], rows: List[Dict]):
    ensure_dir(os.path.dirname(path))

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    os.replace(tmp_path, path)


def append_csv_rows(path: str, columns: List[str], rows: List[Dict]):
    if not rows:
        return

    init_csv_if_needed(path, columns)

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        for row in rows:
            writer.writerow(row)


_write_lock = threading.Lock()


def write_run_error(
    player_id: int,
    player_name: Optional[str],
    stage: str,
    error: str,
):
    row = {
        "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "player_id": player_id,
        "player_name": player_name,
        "stage": stage,
        "error": error,
    }

    with _write_lock:
        append_csv_rows(run_errors_path(), RUN_ERROR_COLUMNS, [row])


def should_skip_player(player_id: int) -> bool:
    if not SKIP_DONE_IF_CSV_EXISTS_AND_NO_ERRORS:
        return False

    games_path = output_games_path(player_id)
    errors_path = output_errors_path(player_id)

    if not os.path.exists(games_path):
        return False

    if csv_has_data_rows(errors_path):
        return False

    return True


def prepare_player_for_retry(player_id: int):
    games_path = output_games_path(player_id)
    tmp_games_path = output_tmp_games_path(player_id)
    errors_path = output_errors_path(player_id)
    tmp_errors_path = output_tmp_errors_path(player_id)

    if OVERWRITE_PLAYER_ON_RETRY and csv_has_data_rows(errors_path):
        for path in [games_path, tmp_games_path, errors_path, tmp_errors_path]:
            if os.path.exists(path):
                os.remove(path)

    for path in [tmp_games_path, tmp_errors_path]:
        if os.path.exists(path):
            os.remove(path)


def format_url_for_log(url: str, params: Optional[dict] = None) -> str:
    if not params:
        return url
    return url + "?" + urlencode(params)


def extract_event_id_from_href(href: str) -> Optional[int]:
    if not href:
        return None

    m = EVENT_ID_RE.search(href)
    if m:
        return int(m.group(1))

    return None


def build_calc_page_url(player_id: int, period: str, rating_type: int) -> str:
    return (
        f"{BASE_URL}calculations.phtml"
        f"?id_number={player_id}&period={period}&rating={rating_type}"
    )


def looks_suspicious_fast(html: str) -> bool:
    if not html:
        return False

    low = html[:3000].lower()
    return any(term in low for term in SUSPICIOUS_TERMS)


# =========================================
# READ FIDE STANDARD RATING LIST
# =========================================

def parse_standard_rating_list(path: str) -> List[Dict]:
    """
    Czyta standard_rating_list.txt pobrany z:
    https://ratings.fide.com/download_lists.phtml

    Uwaga: kolumna Gms dotyczy bieżącej listy/miesiąca,
    więc NIE filtrujemy graczy po Gms=0, jeśli chcemy historię.
    """
    players = []

    header = None
    name_start = None
    fed_start = None
    sex_start = None
    rating_start = None
    gms_start = None
    k_start = None
    bday_start = None
    flag_start = None

    with open(path, "rb") as f:
        for raw in f:
            line = decode_line(raw).rstrip("\r\n")

            if not line.strip():
                continue

            if line.startswith("ID Number"):
                header = line

                name_start = header.find("Name")
                fed_start = header.find("Fed")
                sex_start = header.find("Sex")
                gms_start = header.find("Gms")
                k_start = header.find("K", gms_start + 1)
                bday_start = header.find("B-day")
                flag_start = header.find("Flag")

                rating_match = re.search(r"\b[A-Z]{3}\d{2}\b", header)
                rating_start = rating_match.start() if rating_match else None

                continue

            m = re.match(r"^\s*(\d{4,12})\s+", line)
            if not m:
                continue

            player_id = int(m.group(1))

            if header and all(x is not None and x >= 0 for x in [name_start, fed_start, sex_start]):
                name = clean_text(line[name_start:fed_start])
                fed = clean_text(line[fed_start:sex_start])

                sex = None
                if rating_start is not None and rating_start > sex_start:
                    middle = line[sex_start:rating_start].split()
                    sex = middle[0] if middle else None
                else:
                    sex = clean_text(line[sex_start:sex_start + 3])

                standard_rating = None
                if rating_start is not None and gms_start is not None and gms_start > rating_start:
                    standard_rating = parse_int(line[rating_start:gms_start])

                standard_gms = None
                if gms_start is not None and k_start is not None and k_start > gms_start:
                    standard_gms = parse_int(line[gms_start:k_start])

                birth_year = None
                if bday_start is not None and flag_start is not None and flag_start > bday_start:
                    birth_year = parse_int(line[bday_start:flag_start])
                elif bday_start is not None:
                    birth_year = parse_int(line[bday_start:])

            else:
                rest = line[m.end():].strip()
                tokens = rest.split()

                fed = None
                for tok in tokens:
                    if re.fullmatch(r"[A-Z]{3}", tok):
                        fed = tok
                        break

                name = rest
                if fed:
                    name = rest.split(fed)[0].strip()

                sex = None
                standard_rating = None
                standard_gms = None
                birth_year = None

            players.append({
                "player_id": player_id,
                "player_name_from_list": name,
                "fed": fed,
                "sex": sex,
                "standard_rating_from_list": standard_rating,
                "standard_games_from_list": standard_gms,
                "birth_year": birth_year,
            })

    seen = set()
    unique_players = []

    for p in players:
        pid = p["player_id"]
        if pid in seen:
            continue
        seen.add(pid)
        unique_players.append(p)

    sliced = unique_players[START_OFFSET:]

    if MAX_PLAYERS is not None:
        sliced = sliced[:MAX_PLAYERS]

    return sliced


# =========================================
# HTTP
# =========================================

_thread_local = threading.local()


def create_session() -> requests.Session:
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=PLAYER_WORKERS * 2,
        pool_maxsize=PLAYER_WORKERS * 2,
        max_retries=0,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
        "Connection": "keep-alive",
    })

    return session


def get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = create_session()
    return _thread_local.session


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    backoff_base: Optional[float] = None,
) -> requests.Response:
    if timeout is None:
        timeout = REQUEST_TIMEOUT
    if retries is None:
        retries = REQUEST_RETRIES
    if backoff_base is None:
        backoff_base = REQUEST_BACKOFF

    last_exc = None

    for attempt in range(1, retries + 1):
        if should_stop():
            raise KeyboardInterrupt("Stop requested by user")

        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            if REQUEST_SLEEP_BETWEEN_CALLS > 0:
                time.sleep(REQUEST_SLEEP_BETWEEN_CALLS)

            return response

        except RequestException as e:
            last_exc = e

            if attempt < retries:
                time.sleep(backoff_base * (2 ** (attempt - 1)))

    raise last_exc


# =========================================
# PERIOD INDEX — POST ONLY
# =========================================

def fetch_more_periods_html(session: requests.Session, player_id: int) -> str:
    response = request_with_retries(
        session,
        "POST",
        MORE_PERIODS_URL,
        data={
            "action": "2",
            "plr_id": str(player_id),
        },
        headers={
            "Referer": PROFILE_CALCULATIONS_URL.format(player_id=player_id),
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )

    return response.text


def parse_calc_href(href: str) -> Optional[Dict]:
    if not href:
        return None

    href = html_lib.unescape(str(href).strip())
    full_url = urljoin(BASE_URL, href)

    parsed = urlparse(full_url)
    qs = parse_qs(parsed.query)

    id_number = qs.get("id_number", [None])[0]
    period = qs.get("period", [None])[0]
    rating = qs.get("rating", [None])[0]

    if id_number is None or period is None or rating is None:
        return None

    try:
        id_number_int = int(id_number)
        rating_int = int(rating)
    except ValueError:
        return None

    return {
        "player_id": id_number_int,
        "period": period,
        "rating_type": rating_int,
        "calc_page_url": full_url,
    }


def extract_standard_periods_regex(html: str, player_id: int, source: str) -> List[Dict]:
    rows = []

    for match in HREF_RE.finditer(html or ""):
        href = match.group(1) or match.group(2) or match.group(3)
        href = html_lib.unescape(href)

        if "calculations.phtml" not in href:
            continue

        info = parse_calc_href(href)

        if not info:
            continue

        if info["player_id"] != player_id:
            continue

        if info["rating_type"] != RATING_TYPE:
            continue

        rows.append({
            "player_id": player_id,
            "period": info["period"],
            "rating_type": RATING_TYPE,
            "calc_page_url": info["calc_page_url"],
            "source": source,
        })

    return rows


def build_standard_period_index_for_player(player_id: int) -> List[Dict]:
    session = get_thread_session()

    more_html = fetch_more_periods_html(session, player_id)

    rows = extract_standard_periods_regex(
        html=more_html,
        player_id=player_id,
        source="more_periods_button",
    )

    if not rows:
        return []

    dedup = {}

    for row in rows:
        key = (row["player_id"], row["period"], row["rating_type"])
        if key not in dedup:
            dedup[key] = row

    final_rows = list(dedup.values())
    final_rows.sort(key=lambda r: r["period"], reverse=True)

    return final_rows


# =========================================
# CALC FETCH + PARSE
# =========================================

def fetch_calc_html_network(
    session: requests.Session,
    player_id: int,
    rating_period: str,
    rating_type: int,
) -> str:
    params = {
        "id_number": player_id,
        "rating_period": rating_period,
        "t": rating_type,
    }

    headers = {
        "Referer": build_calc_page_url(player_id, rating_period, rating_type),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "*/*",
    }

    response = request_with_retries(
        session,
        "GET",
        CALC_AJAX_URL,
        params=params,
        headers=headers,
    )

    return response.text


def parse_event_summary_from_table(table) -> Dict:
    summary = {
        "event_rc": None,
        "player_rating": None,
        "event_w": None,
        "event_n": None,
        "event_chg": None,
        "event_k": None,
        "event_k_chg": None,
    }

    rows = table.find_all("tr")

    for row in rows[:4]:
        cells = [clean_text(td.get_text(" ", strip=True)) for td in row.find_all("td")]
        nums = []

        for cell in cells:
            val = parse_float(cell)
            if val is not None:
                nums.append(val)

        if len(nums) >= 7:
            summary["event_rc"] = int(nums[0]) if float(nums[0]).is_integer() else nums[0]
            summary["player_rating"] = int(nums[1]) if float(nums[1]).is_integer() else nums[1]
            summary["event_w"] = nums[2]
            summary["event_n"] = int(nums[3]) if float(nums[3]).is_integer() else nums[3]
            summary["event_chg"] = nums[4]
            summary["event_k"] = int(nums[5]) if float(nums[5]).is_integer() else nums[5]
            summary["event_k_chg"] = nums[6]
            return summary

    return summary


def parse_calc_html_to_game_rows(
    calc_html: str,
    player_meta: Dict,
    period: str,
) -> List[Dict]:
    if not calc_html or clean_text(calc_html) == "":
        return []

    if looks_suspicious_fast(calc_html):
        raise RuntimeError("Suspicious FIDE response")

    soup = BeautifulSoup(calc_html, BS4_PARSER)
    calc_tables = soup.find_all("table", class_="calc_table")

    if not calc_tables:
        return []

    rows_out = []

    player_id = int(player_meta["player_id"])
    player_name = player_meta.get("player_name_from_list")

    for table in calc_tables:
        tournament_div = table.find_previous("div", class_="default_div_full")
        if tournament_div is None:
            continue

        line1 = tournament_div.find("div", class_="rtng_line01")
        line2 = tournament_div.find("div", class_="rtng_line02")

        event_name = None
        event_id = None
        date_from = None
        date_to = None

        if line1:
            a_tag = line1.find("a")
            if a_tag:
                event_name = clean_text(a_tag.get_text(" ", strip=True))
                event_id = extract_event_id_from_href(a_tag.get("href", ""))
            else:
                event_name = clean_text(line1.get_text(" ", strip=True))

        if line2:
            date_spans = line2.find_all("span", class_="dates_span")
            if len(date_spans) >= 1:
                date_from = clean_text(date_spans[0].get_text(" ", strip=True))
            if len(date_spans) >= 2:
                date_to = clean_text(date_spans[1].get_text(" ", strip=True))

        summary = parse_event_summary_from_table(table)

        for row in table.find_all("tr")[2:]:
            cells = row.find_all("td")

            if len(cells) < 10:
                continue

            first_classes = cells[0].get("class") or []
            if "list4" not in first_classes:
                continue

            row_text = clean_text(row.get_text(" ", strip=True))
            if "Rating difference of more than 400" in row_text:
                continue

            color = None
            span = cells[0].find("span")

            if span is not None:
                span_classes = span.get("class") or []
                if "white_note" in span_classes:
                    color = "white"
                elif "black_note" in span_classes:
                    color = "black"

            opponent_name = clean_text(cells[0].get_text(" ", strip=True))

            display_rating_text = clean_text(cells[3].get_text(" ", strip=True))
            display_rating = parse_int(display_rating_text)

            star_flag = False
            if cells[3].find("font") is not None or "*" in display_rating_text:
                star_flag = True

            score = parse_float(cells[5].get_text(" ", strip=True))

            rows_out.append({
                "player_id": player_id,
                "player_name": player_name,
                "fed": player_meta.get("fed"),
                "sex": player_meta.get("sex"),
                "standard_rating_from_list": player_meta.get("standard_rating_from_list"),
                "standard_games_from_list": player_meta.get("standard_games_from_list"),
                "birth_year": player_meta.get("birth_year"),
                "period": period,
                "rating_type": RATING_TYPE,
                "rating_type_name": RATING_TYPE_NAME,
                "event_id": event_id,
                "event_name": event_name,
                "date_from": date_from,
                "date_to": date_to,
                "player_rating": summary.get("player_rating"),
                "event_rc": summary.get("event_rc"),
                "event_w": summary.get("event_w"),
                "event_n": summary.get("event_n"),
                "event_chg": summary.get("event_chg"),
                "event_k": summary.get("event_k"),
                "event_k_chg": summary.get("event_k_chg"),
                "opponent_fide_id": None,
                "opponent_name": opponent_name,
                "opponent_rating": display_rating,
                "display_opponent_rating": display_rating,
                "score": score,
                "color": color,
                "star_400_rule": star_flag,
            })

    return rows_out


# =========================================
# PLAYER SCRAPE
# =========================================

def dedup_game_rows(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return []

    rows = sorted(
        rows,
        key=lambda r: (
            int(r.get("player_id") or 0),
            str(r.get("period") or ""),
            str(r.get("date_from") or ""),
            str(r.get("event_id") or ""),
            str(r.get("opponent_name") or ""),
            str(r.get("score") or ""),
        ),
    )

    seen = set()
    out = []

    for row in rows:
        key = tuple(row.get(col) for col in GAMES_COLUMNS)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def scrape_one_player(player_meta: Dict) -> Dict:
    player_id = int(player_meta["player_id"])
    player_name = player_meta.get("player_name_from_list")

    games_path = output_games_path(player_id)
    tmp_games_path = output_tmp_games_path(player_id)
    errors_path = output_errors_path(player_id)
    tmp_errors_path = output_tmp_errors_path(player_id)

    if should_skip_player(player_id):
        return {
            "player_id": player_id,
            "status": "skipped_existing",
            "games_found": 0,
            "errors": 0,
        }

    prepare_player_for_retry(player_id)

    session = get_thread_session()

    all_game_rows = []
    error_rows = []

    try:
        period_rows = build_standard_period_index_for_player(player_id)

        if not period_rows:
            write_csv_atomic(games_path, tmp_games_path, GAMES_COLUMNS, [])
            if os.path.exists(errors_path):
                os.remove(errors_path)
            return {
                "player_id": player_id,
                "status": "done_no_standard_periods",
                "games_found": 0,
                "errors": 0,
            }

        for period_row in period_rows:
            if should_stop():
                raise KeyboardInterrupt("Stop requested")

            period = str(period_row["period"])

            try:
                calc_html = fetch_calc_html_network(
                    session=session,
                    player_id=player_id,
                    rating_period=period,
                    rating_type=RATING_TYPE,
                )

                rows = parse_calc_html_to_game_rows(
                    calc_html=calc_html,
                    player_meta=player_meta,
                    period=period,
                )

                if rows:
                    all_game_rows.extend(rows)

            except Exception as e:
                error_rows.append({
                    "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "player_id": player_id,
                    "player_name": player_name,
                    "period": period,
                    "stage": "calculation",
                    "error": repr(e),
                })

        final_rows = dedup_game_rows(all_game_rows)

        write_csv_atomic(games_path, tmp_games_path, GAMES_COLUMNS, final_rows)

        if error_rows:
            write_csv_atomic(errors_path, tmp_errors_path, ERROR_COLUMNS, error_rows)
        else:
            for path in [errors_path, tmp_errors_path]:
                if os.path.exists(path):
                    os.remove(path)

        return {
            "player_id": player_id,
            "status": "done",
            "games_found": len(final_rows),
            "errors": len(error_rows),
        }

    except KeyboardInterrupt:
        raise

    except Exception as e:
        error_rows.append({
            "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "player_id": player_id,
            "player_name": player_name,
            "period": None,
            "stage": "player",
            "error": repr(e),
        })

        write_csv_atomic(errors_path, tmp_errors_path, ERROR_COLUMNS, error_rows)

        return {
            "player_id": player_id,
            "status": "failed",
            "games_found": len(all_game_rows),
            "errors": len(error_rows),
        }


# =========================================
# MAIN
# =========================================

def main():
    args = parse_args()
    apply_cli_config(args)

    ensure_dir(OUTPUT_DIR)
    init_csv_if_needed(run_errors_path(), RUN_ERROR_COLUMNS)

    players = parse_standard_rating_list(STANDARD_RATING_LIST_PATH)

    if not players:
        print("[ERROR] No players parsed from standard rating list.")
        return

    to_scrape = []
    skipped_existing = 0

    for p in players:
        pid = int(p["player_id"])
        if should_skip_player(pid):
            skipped_existing += 1
            continue
        to_scrape.append(p)

    print("========== RUN CONFIG ==========")
    print(f"rating list:      {STANDARD_RATING_LIST_PATH}")
    print(f"output dir:       {OUTPUT_DIR}")
    print(f"players parsed:   {len(players)}")
    print(f"start offset:     {START_OFFSET}")
    print(f"max players:      {MAX_PLAYERS}")
    print(f"workers:          {PLAYER_WORKERS}")
    print(f"batch size:       {BATCH_SIZE}")
    print(f"timeout:          {REQUEST_TIMEOUT}s")
    print(f"parser:           {BS4_PARSER}")
    print(f"period index:     POST-only")
    print(f"skipped existing: {skipped_existing}")
    print(f"to scrape:        {len(to_scrape)}")
    print("================================\n")

    total_games = 0
    total_errors = 0
    players_with_games = 0
    skipped_runtime = 0

    start_time = time.time()

    executor = ThreadPoolExecutor(max_workers=PLAYER_WORKERS)
    pending = {}
    next_index = 0

    def submit_more():
        nonlocal next_index

        while (
            not should_stop()
            and next_index < len(to_scrape)
            and len(pending) < BATCH_SIZE
        ):
            player_meta = to_scrape[next_index]
            future = executor.submit(scrape_one_player, player_meta)
            pending[future] = player_meta
            next_index += 1

    try:
        submit_more()

        with tqdm(total=len(to_scrape), desc="players", unit="player") as pbar:
            while pending:
                if should_stop():
                    break

                done, _ = wait(
                    pending.keys(),
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    continue

                for future in done:
                    player_meta = pending.pop(future)
                    player_id = int(player_meta["player_id"])
                    player_name = player_meta.get("player_name_from_list")

                    try:
                        result = future.result()

                        status = result.get("status")
                        games = int(result.get("games_found") or 0)
                        errors = int(result.get("errors") or 0)

                        if status == "skipped_existing":
                            skipped_runtime += 1
                        else:
                            total_games += games
                            total_errors += errors

                            if games > 0:
                                players_with_games += 1

                    except KeyboardInterrupt:
                        raise

                    except Exception as e:
                        total_errors += 1
                        write_run_error(
                            player_id=player_id,
                            player_name=player_name,
                            stage="future",
                            error=repr(e),
                        )

                    pbar.update(1)
                    pbar.set_postfix({
                        "games": total_games,
                        "players_with_games": players_with_games,
                        "errors": total_errors,
                        "queued": len(pending),
                    })

                submit_more()

    except KeyboardInterrupt:
        print("\n[STOPPED] Current completed player CSV files are saved. Unfinished players will retry next run.")

    finally:
        if should_stop():
            for future in pending:
                future.cancel()

            executor.shutdown(wait=False, cancel_futures=True)
            print("[STOPPED] Cancelled queued tasks. Active requests may finish/timeout shortly.")
        else:
            executor.shutdown(wait=True)

    elapsed = time.time() - start_time

    print("\n========== DONE ==========")
    print(f"elapsed seconds:      {elapsed:.2f}")
    print(f"players parsed:       {len(players)}")
    print(f"players scheduled:    {next_index}")
    print(f"players skipped:      {skipped_existing + skipped_runtime}")
    print(f"players with games:   {players_with_games}")
    print(f"games found this run: {total_games}")
    print(f"errors this run:      {total_errors}")
    print(f"output dir:           {OUTPUT_DIR}")
    print("==========================\n")


if __name__ == "__main__":
    main()