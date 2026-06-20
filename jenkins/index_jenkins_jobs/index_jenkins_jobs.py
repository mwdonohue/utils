#!/usr/bin/env python3
"""Index jobs from one or more Jenkins controllers.

Spec
----
* Input       : a list of Jenkins controllers and a maximum folder depth.
* Output      : one JSON file per controller, containing a flat list of jobs.
* Constraints : support Basic auth; support HA controllers by specifying
                ``tree=`` on every request; make exactly one API request per
                controller; do not paginate.

The script calls each controller's Remote Access API once:

    GET {controller}/api/json?tree=jobs[name,fullName,fullDisplayName,url,_class,...]

Nested folder traversal is handled by recursively expanding the ``jobs[...]``
portion of the ``tree`` query according to ``--max-depth``. A max depth of 1
returns only top-level jobs; 2 includes one nested ``jobs`` level, and so on.
By default, the nested Jenkins response is flattened into an array of job
records. Use ``--nested`` to write the raw Jenkins ``{"jobs": [...]}`` shape.
The default flat output is described by ``jenkins-jobs.schema.json``.

Authentication is optional. Jenkins typically uses a username plus API token
with Basic auth. Credentials are read from CLI args first, then these env vars:

    username: JENKINS_USER, JENKINS_USERNAME
    password: JENKINS_API_TOKEN, JENKINS_TOKEN, JENKINS_PASSWORD

Usage
-----
    python index_jenkins_jobs.py https://jenkins.example.com --max-depth 3
    python index_jenkins_jobs.py -i controllers.txt --max-depth 5 -o ./job-indexes
    JENKINS_USER=me JENKINS_API_TOKEN=xxx python index_jenkins_jobs.py ci.local
    python index_jenkins_jobs.py ci.local --max-depth 2 --print-tree
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ModuleNotFoundError:  # pragma: no cover - friendly message, not logic
    sys.exit(
        "This script needs the 'requests' library.\n"
        "Install it with:  python -m pip install requests"
    )

JOB_FIELDS = ("name", "fullName", "fullDisplayName", "url", "_class", "allBuilds[number,url,result,timestamp,actions[parameters[name,value]]]")
USERNAME_ENV_VARS = ("JENKINS_USER", "JENKINS_USERNAME")
PASSWORD_ENV_VARS = ("JENKINS_API_TOKEN", "JENKINS_TOKEN", "JENKINS_PASSWORD")
DEFAULT_TIMEOUT = 30
DEFAULT_OUTPUT_DIR = "."


@dataclass(frozen=True)
class Controller:
    """Normalized Jenkins controller URL details."""

    raw: str
    base_url: str
    api_url: str
    filename_stem: str


def build_jobs_tree(max_depth: int) -> str:
    """Build the recursive Jenkins ``tree=`` value for job indexing.

    ``max_depth`` is job nesting depth, not API path depth:

    * 1 -> jobs[name,fullName,fullDisplayName,url,_class]
    * 2 -> jobs[name,fullName,fullDisplayName,url,_class,jobs[...]]
    """
    if max_depth < 1:
        raise ValueError("--max-depth must be at least 1")

    fields = list(JOB_FIELDS)
    if max_depth > 1:
        fields.append(build_jobs_tree(max_depth - 1))
    return f"jobs[{','.join(fields)}]"


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
    """Normalize a user-supplied controller into base/API URLs and a file stem."""
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
            "script builds the Jenkins API query itself"
        )

    path = parts.path.rstrip("/")
    if path.endswith("/api/json"):
        api_path = path
        base_path = path[: -len("/api/json")] or ""
    else:
        base_path = path
        api_path = (path + "/api/json") if path else "/api/json"

    base_url = urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))
    api_url = urlunsplit((parts.scheme, parts.netloc, api_path, "", ""))
    return Controller(
        raw=raw_url,
        base_url=base_url,
        api_url=api_url,
        filename_stem=_safe_filename_stem(urlsplit(base_url)),
    )


def parse_controller_text(text: str) -> list[str]:
    """Parse controller URLs from text, one per line, with # comments."""
    controllers: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            controllers.append(line)
    return controllers


def load_controllers(args: argparse.Namespace) -> list[str]:
    """Gather controllers from args, --input files, and/or stdin; dedupe."""
    collected: list[str] = list(args.controllers)
    sources = list(args.input)

    if "-" in sources or (not collected and not sources and not sys.stdin.isatty()):
        collected += parse_controller_text(sys.stdin.read())
        sources = [source for source in sources if source != "-"]

    for path in sources:
        try:
            with open(path, encoding="utf-8") as fh:
                collected += parse_controller_text(fh.read())
        except OSError as exc:
            raise SystemExit(f"Cannot read controller file '{path}': {exc}")

    seen: set[str] = set()
    unique: list[str] = []
    for controller in collected:
        if controller not in seen:
            seen.add(controller)
            unique.append(controller)
    return unique


def output_path_for(
    controller: Controller,
    output_dir: str,
    used_stems: dict[str, int],
) -> str:
    """Return a collision-safe output path for one controller."""
    count = used_stems.get(controller.filename_stem, 0) + 1
    used_stems[controller.filename_stem] = count

    suffix = "" if count == 1 else f"-{count}"
    filename = f"{controller.filename_stem}{suffix}-jobs.json"
    return os.path.abspath(os.path.join(output_dir, filename))


def fetch_job_index(
    session: requests.Session,
    controller: Controller,
    tree: str,
    auth: HTTPBasicAuth | None,
    timeout: int,
) -> dict[str, Any]:
    """Fetch one controller's job index with one HTTP request."""
    resp = session.get(
        controller.api_url,
        params={"tree": tree},
        auth=auth,
        timeout=timeout,
        allow_redirects=False,
    )

    if 300 <= resp.status_code < 400:
        location = resp.headers.get("Location", "(missing Location header)")
        raise RuntimeError(
            f"{controller.base_url} redirected to {location}. Provide the final "
            "controller URL so the one-call contract is preserved."
        )
    if resp.status_code == 401:
        raise RuntimeError(
            f"{controller.base_url} rejected the credentials (HTTP 401)."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            f"{controller.base_url} denied access (HTTP 403). Check the Jenkins "
            "permissions for the supplied user/token."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"{controller.api_url} was not found (HTTP 404). Check the controller "
            "URL, including any Jenkins context path."
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{controller.base_url} returned {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        content_type = resp.headers.get("Content-Type", "unknown content type")
        raise RuntimeError(
            f"{controller.base_url} did not return JSON ({content_type})."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"{controller.base_url} returned a non-object JSON value.")
    if "jobs" not in data:
        data["jobs"] = []
    return data


def flatten_jobs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Jenkins' nested jobs tree into a list of requested job fields."""
    flattened: list[dict[str, Any]] = []

    def visit(jobs: Any) -> None:
        if not isinstance(jobs, list):
            return
        for job in jobs:
            if not isinstance(job, dict):
                continue
            flattened.append({field: job.get(field) for field in JOB_FIELDS})
            visit(job.get("jobs"))

    visit(data.get("jobs"))
    return flattened


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
        description="Index Jenkins jobs from one or more controllers using one "
        "tree-filtered API call per controller.",
    )
    parser.add_argument(
        "controllers",
        nargs="*",
        help="Jenkins controller base URLs. If a scheme is omitted, https:// is used.",
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        default=[],
        metavar="FILE",
        help="Read controller URLs from FILE, one per line. Repeatable. Use '-' "
        "for stdin.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        required=True,
        help="Maximum job/folder nesting depth to request. 1 means top-level jobs.",
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
        help=f"HTTP timeout in seconds for each controller (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON instead of indented, human-readable JSON.",
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Write Jenkins' nested {'jobs': [...]} response instead of a flat "
        "job array.",
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

    controllers = load_controllers(args)
    if not controllers:
        raise SystemExit(
            "No Jenkins controllers provided. Pass URLs as arguments, via "
            "-i/--input FILE, or on stdin. See --help."
        )

    username = args.username or _first_env(USERNAME_ENV_VARS)
    password = args.password or _first_env(PASSWORD_ENV_VARS)
    if bool(username) != bool(password):
        raise SystemExit(
            "Basic auth needs both username and password/token. Provide both "
            "via CLI args or Jenkins env vars."
        )
    auth = HTTPBasicAuth(username, password) if username and password else None

    tree = build_jobs_tree(args.max_depth)
    if args.print_tree:
        print(f"tree={tree}", file=sys.stderr)
    if auth is None:
        print("No Basic auth credentials provided.", file=sys.stderr)

    session = requests.Session()
    session.headers.update({"User-Agent": "index-jenkins-jobs-script"})

    normalized: list[Controller] = []
    for raw_controller in controllers:
        try:
            normalized.append(normalize_controller(raw_controller))
        except ValueError as exc:
            print(f"[FAIL] {raw_controller}: {exc}", file=sys.stderr)

    failures = len(controllers) - len(normalized)
    output_stems: dict[str, int] = {}
    started = time.perf_counter()

    for controller in normalized:
        path = output_path_for(controller, args.output_dir, output_stems)
        try:
            data = fetch_job_index(
                session=session,
                controller=controller,
                tree=tree,
                auth=auth,
                timeout=args.timeout,
            )
            output_data: Any = data if args.nested else flatten_jobs(data)
            write_json(path, output_data, compact=args.compact)
        except (OSError, requests.RequestException, RuntimeError) as exc:
            failures += 1
            print(f"[FAIL] {controller.base_url}: {exc}", file=sys.stderr)
            continue

        if args.nested:
            jobs = data.get("jobs")
            job_count = len(jobs) if isinstance(jobs, list) else 0
            job_label = "top-level job(s)"
        else:
            job_count = len(output_data)
            job_label = "job(s)"
        print(
            f"[ OK ] {controller.base_url}: wrote {path} "
            f"({job_count} {job_label})",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - started
    succeeded = len(normalized) - (failures - (len(controllers) - len(normalized)))
    print(
        f"Done in {elapsed:.2f}s: {succeeded} succeeded, {failures} failed.",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
