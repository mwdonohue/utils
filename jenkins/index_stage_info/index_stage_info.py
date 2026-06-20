#!/usr/bin/env python3
"""Index Pipeline stage information from Jenkins controllers.

Spec
----
* Input       : a JSON map of Jenkins controller URL to a deduped list of jobs.
* Output      : one JSON file per controller, containing per-job run/stage data.
* Constraints : support Basic auth; support HA controllers by specifying
                ``tree=`` on every request; make one API request per job; do
                not paginate; construct the ``tree=`` query recursively from
                ``--max-depth``.

The script calls each job's Pipeline REST API once:

    GET {controller}/job/{folder}/job/{job}/wfapi/runs?tree=...

The ``wfapi/runs`` endpoint is job-scoped, so a controller with N jobs requires
N requests. Results are grouped into one controller output file. The input map
may contain job names as strings or job objects from ``index_jenkins_jobs.py``;
objects are read using ``fullName``, then ``url``, then ``name``.

Jobs for the same controller are fetched concurrently with a bounded worker
pool. The output stays deterministic: jobs are written in the same order they
appear in the normalized input, regardless of request completion order.

Authentication is optional. Jenkins typically uses a username plus API token
with Basic auth. Credentials are read from CLI args first, then these env vars:

    username: JENKINS_USER, JENKINS_USERNAME
    password: JENKINS_API_TOKEN, JENKINS_TOKEN, JENKINS_PASSWORD

Usage
-----
    python index_stage_info.py controller-jobs.json --max-depth 2
    python index_stage_info.py - --max-depth 1 -o ./stage-indexes
    python index_stage_info.py jobs.json --max-depth 2 --workers 16
    python index_stage_info.py one-controller-jobs.json --controller ci.local --max-depth 2
    JENKINS_USER=me JENKINS_API_TOKEN=xxx python index_stage_info.py jobs.json --max-depth 2
    python index_stage_info.py jobs.json --max-depth 2 --print-tree

Example input
-------------
    {
      "https://jenkins.example.com": [
        "folder/pipeline-job",
        {"fullName": "another-pipeline", "url": "https://jenkins.example.com/job/another-pipeline/"}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlsplit, urlunsplit

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.auth import HTTPBasicAuth
    from urllib3.util.retry import Retry
except ModuleNotFoundError:  # pragma: no cover - friendly message, not logic
    sys.exit(
        "This script needs the 'requests' library.\n"
        "Install it with:  python -m pip install requests"
    )

RUN_FIELDS = (
    "_links[*]",
    "id",
    "name",
    "status",
    "startTimeMillis",
    "endTimeMillis",
    "durationMillis",
    "queueDurationMillis",
    "pauseDurationMillis",
)
STAGE_FIELDS = (
    "_links[*]",
    "id",
    "name",
    "execNode",
    "status",
    "startTimeMillis",
    "durationMillis",
    "pauseDurationMillis",
)
FLOW_NODE_FIELDS = (
    "_links[*]",
    "id",
    "name",
    "status",
    "startTimeMillis",
    "durationMillis",
    "pauseDurationMillis",
    "parentNodes",
    "error[message,type]",
)
USERNAME_ENV_VARS = ("JENKINS_USER", "JENKINS_USERNAME")
PASSWORD_ENV_VARS = ("JENKINS_API_TOKEN", "JENKINS_TOKEN", "JENKINS_PASSWORD")
DEFAULT_TIMEOUT = 30
DEFAULT_OUTPUT_DIR = "."
DEFAULT_WORKERS = 8
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 0.5


@dataclass(frozen=True)
class Controller:
    """Normalized Jenkins controller URL details."""

    raw: str
    base_url: str
    filename_stem: str


@dataclass(frozen=True)
class JobRef:
    """A normalized job reference for one Jenkins controller."""

    full_name: str
    job_url: str
    runs_url: str


@dataclass(frozen=True)
class JobFetchResult:
    """Fetched stage information for one Jenkins job."""

    index: int
    job: JobRef
    runs: list[Any]


def build_runs_tree(max_depth: int) -> str:
    """Build the recursive Jenkins ``tree=`` value for stage indexing.

    ``max_depth`` controls how deeply stage internals are requested:

    * 1 -> run fields plus stage summary fields.
    * 2 -> also include each stage's ``stageFlowNodes``.
    * 3+ -> recursively request nested ``stageFlowNodes`` if the controller
      exposes them.
    """
    if max_depth < 1:
        raise ValueError("--max-depth must be at least 1")

    fields = list(RUN_FIELDS)
    fields.append(f"stages[{build_stage_tree(max_depth)}]")
    return ",".join(fields)


def build_stage_tree(max_depth: int) -> str:
    fields = list(STAGE_FIELDS)
    if max_depth > 1:
        fields.append(f"stageFlowNodes[{build_flow_node_tree(max_depth - 1)}]")
    return ",".join(fields)


def build_flow_node_tree(max_depth: int) -> str:
    fields = list(FLOW_NODE_FIELDS)
    if max_depth > 1:
        fields.append(f"stageFlowNodes[{build_flow_node_tree(max_depth - 1)}]")
    return ",".join(fields)


def _first_env(names: tuple[str, ...]) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def _safe_filename_stem(parts: Any) -> str:
    host = parts.hostname or "jenkins"
    if parts.port:
        host = f"{host}-{parts.port}"
    path = parts.path.strip("/")
    raw = host if not path else f"{host}-{path.replace('/', '-')}"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-_")
    return stem or "jenkins"


def normalize_controller(raw_url: str) -> Controller:
    """Normalize a controller URL into a base URL and file stem."""
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("empty controller URL")
    if "://" not in raw_url:
        raw_url = "https://" + raw_url

    parts = urlsplit(raw_url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme '{parts.scheme}' for {raw_url!r}")
    if not parts.netloc:
        raise ValueError(f"controller URL must include a host: {raw_url!r}")
    if parts.username or parts.password:
        raise ValueError(
            "do not put credentials in the controller URL; use --username and "
            "--password or Jenkins env vars instead"
        )
    if parts.query or parts.fragment:
        raise ValueError(
            "controller URL must not include a query string or fragment; this "
            "script builds Jenkins API queries itself"
        )

    path = parts.path.rstrip("/")
    if path.endswith("/api/json"):
        path = path[: -len("/api/json")] or ""

    base_url = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return Controller(
        raw=raw_url,
        base_url=base_url,
        filename_stem=_safe_filename_stem(urlsplit(base_url)),
    )


def parse_job_name_from_url(raw_url: str) -> str | None:
    """Extract a Jenkins full job name from a ``.../job/name/job/name`` URL."""
    parts = urlsplit(raw_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None

    path_parts = [unquote(part) for part in parts.path.strip("/").split("/") if part]
    names: list[str] = []
    index = 0
    while index < len(path_parts):
        if path_parts[index] == "job" and index + 1 < len(path_parts):
            names.append(path_parts[index + 1])
            index += 2
        else:
            index += 1
    return "/".join(names) if names else None


def normalize_job_name(raw_name: str) -> str:
    """Normalize a Jenkins full job name supplied as text."""
    raw_name = raw_name.strip()
    if not raw_name:
        raise ValueError("empty job name")

    parsed_from_url = parse_job_name_from_url(raw_name)
    if parsed_from_url:
        raw_name = parsed_from_url

    parts = [part.strip() for part in raw_name.strip("/").split("/") if part.strip()]
    if not parts:
        raise ValueError("empty job name")
    return "/".join(parts)


def iter_job_names(value: Any) -> Iterable[str]:
    """Yield job names from supported map values, including nested job objects."""
    if isinstance(value, str):
        yield normalize_job_name(value)
        return

    if isinstance(value, dict):
        candidate = value.get("fullName")
        if isinstance(candidate, str) and candidate.strip():
            yield normalize_job_name(candidate)
        elif isinstance(value.get("url"), str):
            parsed = parse_job_name_from_url(value["url"])
            if parsed:
                yield normalize_job_name(parsed)
        elif isinstance(value.get("name"), str) and value["name"].strip():
            yield normalize_job_name(value["name"])

        children = value.get("jobs")
        if isinstance(children, list):
            yield from iter_job_names(children)
        return

    if isinstance(value, list):
        for item in value:
            yield from iter_job_names(item)
        return

    raise ValueError(f"unsupported job entry type: {type(value).__name__}")


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def load_controller_jobs(path: str, controller_override: str | None) -> dict[str, list[str]]:
    """Read and normalize the input JSON controller-to-jobs map."""
    try:
        if path == "-":
            data = json.load(sys.stdin)
        else:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"Cannot read input file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Input file '{path}' is not valid JSON: {exc}") from exc

    if controller_override:
        jobs = data.get("jobs") if isinstance(data, dict) and "jobs" in data else data
        try:
            names = dedupe_preserving_order(iter_job_names(jobs))
        except ValueError as exc:
            raise SystemExit(f"Invalid jobs input for {controller_override!r}: {exc}") from exc
        return {controller_override: names}

    if isinstance(data, dict) and set(data) == {"jobs"}:
        raise SystemExit(
            "Input looks like a single-controller Jenkins jobs export "
            "({'jobs': [...]}). Pass --controller CONTROLLER_URL, or wrap the "
            "jobs list as {CONTROLLER_URL: [...]}."
        )

    if not isinstance(data, dict):
        raise SystemExit(
            "Input JSON must be an object mapping controller URL to jobs. "
            "For a single controller's jobs array, pass --controller CONTROLLER_URL."
        )

    normalized: dict[str, list[str]] = {}
    for controller, jobs in data.items():
        if not isinstance(controller, str) or not controller.strip():
            raise SystemExit("Every input map key must be a non-empty controller URL.")
        try:
            names = dedupe_preserving_order(iter_job_names(jobs))
        except ValueError as exc:
            raise SystemExit(f"Invalid jobs list for {controller!r}: {exc}") from exc
        normalized[controller] = names
    return normalized


def job_urls(controller: Controller, full_name: str) -> JobRef:
    """Build canonical job and wfapi/runs URLs for a normalized job name."""
    parts = urlsplit(controller.base_url)
    path = parts.path.rstrip("/")
    for segment in full_name.split("/"):
        path += "/job/" + quote(segment, safe="")
    job_url = urlunsplit((parts.scheme, parts.netloc, path + "/", "", ""))
    runs_url = urlunsplit((parts.scheme, parts.netloc, path + "/wfapi/runs", "", ""))
    return JobRef(full_name=full_name, job_url=job_url, runs_url=runs_url)


def output_path_for(
    controller: Controller,
    output_dir: str,
    used_stems: dict[str, int],
) -> str:
    """Return a collision-safe output path for one controller."""
    count = used_stems.get(controller.filename_stem, 0) + 1
    used_stems[controller.filename_stem] = count

    suffix = "" if count == 1 else f"-{count}"
    filename = f"{controller.filename_stem}{suffix}-stage-info.json"
    return os.path.abspath(os.path.join(output_dir, filename))


def build_session(
    pool_size: int,
    retries: int,
    retry_backoff: float,
) -> requests.Session:
    """Create a requests session with keep-alive, pooling, and GET retries."""
    session = requests.Session()
    session.headers.update({"User-Agent": "index-stage-info-script"})

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=retry_backoff,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_stage_info(
    session: requests.Session,
    job: JobRef,
    tree: str,
    auth: HTTPBasicAuth | None,
    timeout: int,
    full_stages: bool,
    since: str | None,
) -> list[Any]:
    """Fetch one job's run/stage data with one HTTP request."""
    params: dict[str, Any] = {"tree": tree}
    if full_stages:
        params["fullStages"] = "true"
    if since:
        params["since"] = since

    resp = session.get(
        job.runs_url,
        params=params,
        auth=auth,
        timeout=timeout,
        allow_redirects=False,
    )

    if 300 <= resp.status_code < 400:
        location = resp.headers.get("Location", "(missing Location header)")
        raise RuntimeError(
            f"{job.runs_url} redirected to {location}. Provide the final "
            "controller URL so the one-call contract is preserved."
        )
    if resp.status_code == 401:
        raise RuntimeError(f"{job.runs_url} rejected the credentials (HTTP 401).")
    if resp.status_code == 403:
        raise RuntimeError(
            f"{job.runs_url} denied access (HTTP 403). Check the Jenkins "
            "permissions for the supplied user/token."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"{job.runs_url} was not found (HTTP 404). Check that the job is a "
            "Pipeline job and that the Pipeline REST API plugin is installed."
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{job.runs_url} returned {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        content_type = resp.headers.get("Content-Type", "unknown content type")
        raise RuntimeError(f"{job.runs_url} did not return JSON ({content_type}).") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"{job.runs_url} returned a non-array JSON value.")
    return data


def fetch_controller_jobs(
    session: requests.Session,
    controller: Controller,
    job_names: list[str],
    tree: str,
    auth: HTTPBasicAuth | None,
    timeout: int,
    full_stages: bool,
    since: str | None,
    workers: int,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch all jobs for one controller concurrently, preserving output order."""
    if not job_names:
        return [], 0

    max_workers = min(workers, len(job_names))
    failures = 0
    results: dict[int, dict[str, Any]] = {}

    def fetch_one(index: int, name: str) -> JobFetchResult:
        job = job_urls(controller, name)
        runs = fetch_stage_info(
            session=session,
            job=job,
            tree=tree,
            auth=auth,
            timeout=timeout,
            full_stages=full_stages,
            since=since,
        )
        return JobFetchResult(index=index, job=job, runs=runs)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(fetch_one, index, name): name
            for index, name in enumerate(job_names)
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
            except (requests.RequestException, RuntimeError) as exc:
                failures += 1
                print(f"[FAIL] {controller.base_url} :: {name}: {exc}", file=sys.stderr)
                continue

            results[result.index] = {
                "name": result.job.full_name,
                "url": result.job.job_url,
                "runs": result.runs,
            }
            print(
                f"[ OK ] {controller.base_url} :: {name}: "
                f"fetched {len(result.runs)} run(s)",
                file=sys.stderr,
            )

    ordered = [results[index] for index in range(len(job_names)) if index in results]
    return ordered, failures


def write_json(path: str, data: Any, compact: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if compact:
            json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Jenkins Pipeline run/stage details from a controller-to-jobs "
        "JSON map using one tree-filtered wfapi/runs call per job.",
    )
    parser.add_argument(
        "input",
        help="JSON file mapping controller URL to job list. Use '-' for stdin.",
    )
    parser.add_argument(
        "--controller",
        help="Controller URL to use when INPUT is a single controller job export "
        "such as {'jobs': [...]} or a bare jobs array.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        required=True,
        help="Stage detail depth for the generated tree query. 1 means stage summaries; "
        "2 includes stageFlowNodes.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Directory for per-controller JSON files (default: {DEFAULT_OUTPUT_DIR!r}).",
    )
    parser.add_argument(
        "--username",
        help="Jenkins Basic auth username. Falls back to $JENKINS_USER, then "
        "$JENKINS_USERNAME.",
    )
    parser.add_argument(
        "--password",
        help="Jenkins Basic auth password or API token. Falls back to "
        "$JENKINS_API_TOKEN, $JENKINS_TOKEN, then $JENKINS_PASSWORD.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds for each job (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent job fetches per controller (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries for transient GET failures (default: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help=f"Retry backoff factor in seconds (default: {DEFAULT_RETRY_BACKOFF}).",
    )
    parser.add_argument(
        "--since",
        help="Optional wfapi/runs 'since' value, passed with the same one request.",
    )
    parser.add_argument(
        "--no-full-stages",
        action="store_true",
        help="Do not pass fullStages=true. By default, detailed stageFlowNodes are requested.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON instead of indented, human-readable JSON.",
    )
    parser.add_argument(
        "--print-tree",
        action="store_true",
        help="Print the generated Jenkins tree query to stderr before fetching.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_depth < 1:
        raise SystemExit("--max-depth must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1 second.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.retries < 0:
        raise SystemExit("--retries must be 0 or greater.")
    if args.retry_backoff < 0:
        raise SystemExit("--retry-backoff must be 0 or greater.")

    controller_jobs = load_controller_jobs(args.input, args.controller)
    if not controller_jobs:
        raise SystemExit("Input JSON did not contain any controllers.")

    username = args.username or _first_env(USERNAME_ENV_VARS)
    password = args.password or _first_env(PASSWORD_ENV_VARS)
    if bool(username) != bool(password):
        raise SystemExit(
            "Basic auth needs both username and password/token. Provide both "
            "via CLI args or Jenkins env vars."
        )
    auth = HTTPBasicAuth(username, password) if username and password else None

    tree = build_runs_tree(args.max_depth)
    if args.print_tree:
        print(f"tree={tree}", file=sys.stderr)
    if auth is None:
        print("No Basic auth credentials provided.", file=sys.stderr)

    session = build_session(
        pool_size=args.workers,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )

    failures = 0
    output_stems: dict[str, int] = {}
    started = time.perf_counter()

    for raw_controller, job_names in controller_jobs.items():
        try:
            controller = normalize_controller(raw_controller)
        except ValueError as exc:
            failures += 1
            print(f"[FAIL] {raw_controller}: {exc}", file=sys.stderr)
            continue

        path = output_path_for(controller, args.output_dir, output_stems)
        jobs_output, controller_failures = fetch_controller_jobs(
            session=session,
            controller=controller,
            job_names=job_names,
            tree=tree,
            auth=auth,
            timeout=args.timeout,
            full_stages=not args.no_full_stages,
            since=args.since,
            workers=args.workers,
        )
        failures += controller_failures

        try:
            write_json(path, jobs_output, compact=args.compact)
        except OSError as exc:
            failures += 1
            print(f"[FAIL] {controller.base_url}: cannot write {path}: {exc}", file=sys.stderr)
            continue

        print(
            f"[ OK ] {controller.base_url}: wrote {path} "
            f"({len(jobs_output)} job(s), {controller_failures} failed)",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed:.2f}s: {failures} failed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
