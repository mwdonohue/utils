#!/usr/bin/env python3
"""Index Jira tickets with a single search API call.

Spec
----
* Input       : a Jira project key or a fix version.
* Output      : one JSON file per returned ticket.
* Constraints : support Basic auth; make exactly one Jira API request; do not
                paginate, retry, follow redirects, or call per-ticket APIs.

The script sends one JQL search request and writes each returned issue object to
its own file:

    POST {jira}/rest/api/{version}/search

For Jira Cloud tenants that have moved to enhanced JQL search, use:

    --api-version 3 --search-mode enhanced

which calls:

    POST {jira}/rest/api/3/search/jql

By default, the request asks Jira for every field available through search
(``fields=*all``) and common expansions (names, schema, operations, editmeta,
changelog, versionedRepresentations, transitions, renderedFields). Some Jira
fields, such as comments, worklogs, and changelog entries, can still be
internally paged by Jira inside the issue payload. This script preserves the
one-call contract and writes whatever Jira returns in that one response.

If Jira or a gateway times out on that large one-call payload, keep the
one-call contract but shrink the response with ``--max-results``,
``--no-expand``, or ``--fields key,summary,status,fixVersions``.

Authentication is optional. Jira Cloud typically uses an email address plus API
token with Basic auth. Credentials are read from CLI args first, then env vars:

    base URL            : JIRA_BASE_URL, JIRA_URL
    username            : JIRA_USER, JIRA_USERNAME, ATLASSIAN_EMAIL
    password            : JIRA_API_TOKEN, JIRA_TOKEN, JIRA_PASSWORD,
                          ATLASSIAN_API_TOKEN
    JKS truststore path : JIRA_TRUSTSTORE, JIRA_TRUSTSTORE_PATH
    truststore password : JIRA_TRUSTSTORE_PASSWORD, TRUSTSTORE_PASSWORD

If ``--truststore`` points at a JKS truststore, the script uses ``keytool`` to
export the trusted certificates into a temporary PEM bundle, passes that bundle
to ``requests`` as the TLS verifier, and deletes the bundle after the one Jira
request finishes.

Usage
-----
    python index_jira_tickets.py --base-url https://jira.example.com --project ABC
    python index_jira_tickets.py --base-url https://jira.example.com --fix-version 1.2.3
    python index_jira_tickets.py ABC -o ./ticket-index
    JIRA_BASE_URL=https://example.atlassian.net JIRA_USER=me@example.com \
      JIRA_API_TOKEN=xxx python index_jira_tickets.py --api-version 3 \
      --search-mode enhanced --project ABC
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ModuleNotFoundError:  # pragma: no cover - friendly message, not logic
    sys.exit(
        "This script needs the 'requests' library.\n"
        "Install it with:  python -m pip install requests"
    )

BASE_URL_ENV_VARS = ("JIRA_BASE_URL", "JIRA_URL")
USERNAME_ENV_VARS = ("JIRA_USER", "JIRA_USERNAME", "ATLASSIAN_EMAIL")
PASSWORD_ENV_VARS = (
    "JIRA_API_TOKEN",
    "JIRA_TOKEN",
    "JIRA_PASSWORD",
    "ATLASSIAN_API_TOKEN",
)
TRUSTSTORE_ENV_VARS = ("JIRA_TRUSTSTORE", "JIRA_TRUSTSTORE_PATH")
TRUSTSTORE_PASSWORD_ENV_VARS = ("JIRA_TRUSTSTORE_PASSWORD", "TRUSTSTORE_PASSWORD")
KEYTOOL_ENV_VARS = ("KEYTOOL", "JAVA_KEYTOOL")
DEFAULT_API_VERSION = 2
DEFAULT_MAX_RESULTS = 1000
DEFAULT_OUTPUT_DIR = "."
DEFAULT_SEARCH_MODE = "classic"
DEFAULT_TRUSTSTORE_TYPE = "JKS"
DEFAULT_TIMEOUT = 30
DEFAULT_FIELDS = ("*all",)
DEFAULT_EXPANDS = (
    "names",
    "schema",
    "operations",
    "editmeta",
    "changelog",
    "versionedRepresentations",
    "transitions",
    "renderedFields",
)
PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
CERTIFICATE_RE = re.compile(
    r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.DOTALL,
)


@dataclass(frozen=True)
class JiraSite:
    """Normalized Jira site details."""

    raw: str
    base_url: str
    filename_stem: str


def _first_env(names: tuple[str, ...]) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def _safe_filename_stem(raw: str, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-_")
    return stem or fallback


def normalize_jira_site(raw_url: str) -> JiraSite:
    """Normalize a Jira URL into a base URL and file stem."""
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("empty Jira base URL")
    if "://" not in raw_url:
        raw_url = "https://" + raw_url

    parts = urlsplit(raw_url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme '{parts.scheme}' for {raw_url!r}")
    if not parts.netloc:
        raise ValueError(f"Jira base URL must include a host: {raw_url!r}")
    if parts.username or parts.password:
        raise ValueError(
            "do not put credentials in the Jira URL; use --username and "
            "--password or Jira env vars instead"
        )
    if parts.query or parts.fragment:
        raise ValueError("Jira base URL must not include a query string or fragment")

    path = parts.path.rstrip("/")
    rest_index = path.find("/rest/api/")
    if rest_index >= 0:
        path = path[:rest_index].rstrip("/")

    base_url = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    host = parts.hostname or "jira"
    if parts.port:
        host = f"{host}-{parts.port}"
    path_stem = path.strip("/").replace("/", "-")
    raw_stem = host if not path_stem else f"{host}-{path_stem}"
    return JiraSite(
        raw=raw_url,
        base_url=base_url,
        filename_stem=_safe_filename_stem(raw_stem, "jira"),
    )


def jql_quote(value: str) -> str:
    """Return a Jira JQL quoted string literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_jql(project: str | None, fix_version: str | None, target: str | None) -> str:
    """Build JQL from explicit flags or an inferred positional target."""
    if sum(bool(value) for value in (project, fix_version, target)) != 1:
        raise ValueError(
            "provide exactly one input: --project KEY, --fix-version VERSION, "
            "or a positional target"
        )

    if target:
        target = target.strip()
        if not target:
            raise ValueError("target cannot be empty")
        if PROJECT_KEY_RE.fullmatch(target):
            project = target
        else:
            fix_version = target

    if project:
        project = project.strip()
        if not project:
            raise ValueError("project key cannot be empty")
        return f"project = {jql_quote(project)} ORDER BY key ASC"

    assert fix_version is not None
    fix_version = fix_version.strip()
    if not fix_version:
        raise ValueError("fix version cannot be empty")
    return f"fixVersion = {jql_quote(fix_version)} ORDER BY key ASC"


def search_url(site: JiraSite, api_version: int, search_mode: str) -> str:
    path = f"/rest/api/{api_version}/search"
    if search_mode == "enhanced":
        path += "/jql"

    parts = urlsplit(site.base_url)
    base_path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, base_path + path, "", ""))


def request_body(
    jql: str,
    *,
    max_results: int,
    search_mode: str,
    fields: tuple[str, ...],
    expand: tuple[str, ...],
) -> dict[str, Any]:
    """Build the search request body for classic or enhanced Jira search."""
    body: dict[str, Any] = {
        "jql": jql,
        "maxResults": max_results,
        "fields": list(fields),
    }
    if search_mode == "classic":
        body["startAt"] = 0
        body["validateQuery"] = True
        if expand:
            body["expand"] = list(expand)
    else:
        if expand:
            body["expand"] = ",".join(expand)
    return body


def build_auth(username: str | None, password: str | None) -> HTTPBasicAuth | None:
    if bool(username) != bool(password):
        raise ValueError(
            "Basic auth needs both username and password/token. Provide both "
            "via CLI args or Jira env vars."
        )
    return HTTPBasicAuth(username, password) if username and password else None


def export_truststore_to_pem(
    truststore_path: str,
    truststore_password: str,
    truststore_type: str,
    keytool: str,
) -> str:
    """Export a Java truststore's certs into a temporary PEM bundle."""
    if not os.path.isfile(truststore_path):
        raise ValueError(f"truststore does not exist or is not a file: {truststore_path}")
    if not truststore_password:
        raise ValueError(
            "a truststore password is required. Use --truststore-password or "
            "set JIRA_TRUSTSTORE_PASSWORD."
        )

    cmd = [
        keytool,
        "-list",
        "-rfc",
        "-keystore",
        truststore_path,
        "-storepass",
        truststore_password,
        "-storetype",
        truststore_type,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"cannot find keytool executable {keytool!r}. Install a JDK/JRE or "
            "pass --keytool PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "(no stderr)"
        raise RuntimeError(
            f"keytool could not read truststore {truststore_path!r}: {stderr}"
        ) from exc

    certs = CERTIFICATE_RE.findall(result.stdout)
    if not certs:
        raise RuntimeError(
            f"keytool read {truststore_path!r}, but did not output any PEM certificates."
        )

    fd, pem_path = tempfile.mkstemp(prefix="jira-truststore-", suffix=".pem")
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as fh:
            fh.write("\n".join(certs))
            fh.write("\n")
    except Exception:
        os.unlink(pem_path)
        raise

    return pem_path


@contextmanager
def tls_verify_bundle(
    truststore_path: str | None,
    truststore_password: str | None,
    truststore_type: str,
    keytool: str,
) -> Iterator[bool | str]:
    """Yield the requests ``verify`` value, converting JKS to PEM if needed."""
    if not truststore_path:
        yield True
        return

    pem_path = export_truststore_to_pem(
        truststore_path=truststore_path,
        truststore_password=truststore_password or "",
        truststore_type=truststore_type,
        keytool=keytool,
    )
    try:
        yield pem_path
    finally:
        try:
            os.unlink(pem_path)
        except OSError:
            pass


def fetch_issues_once(
    site: JiraSite,
    *,
    api_version: int,
    search_mode: str,
    jql: str,
    max_results: int,
    fields: tuple[str, ...],
    expand: tuple[str, ...],
    auth: HTTPBasicAuth | None,
    timeout: int,
    verify: bool | str,
) -> dict[str, Any]:
    """Fetch tickets with exactly one Jira API request."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "index-jira-tickets-script",
        }
    )
    url = search_url(site, api_version, search_mode)
    resp = session.post(
        url,
        data=json.dumps(
            request_body(
                jql,
                max_results=max_results,
                search_mode=search_mode,
                fields=fields,
                expand=expand,
            )
        ),
        auth=auth,
        timeout=timeout,
        verify=verify,
        allow_redirects=False,
    )

    if 300 <= resp.status_code < 400:
        location = resp.headers.get("Location", "(missing Location header)")
        raise RuntimeError(
            f"{url} redirected to {location}. Provide the final Jira URL so "
            "the one-call contract is preserved."
        )
    if resp.status_code == 400:
        raise RuntimeError(f"Jira rejected the JQL or request body (HTTP 400): {resp.text}")
    if resp.status_code == 401:
        raise RuntimeError("Jira rejected the credentials (HTTP 401).")
    if resp.status_code == 403:
        raise RuntimeError(
            "Jira denied access (HTTP 403). Check Browse Projects permission "
            "for the supplied user/token."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"{url} was not found (HTTP 404). Check --api-version and "
            "--search-mode for this Jira site."
        )
    if resp.status_code == 504:
        raise RuntimeError(
            "Jira or an upstream gateway timed out (HTTP 504). The request is "
            "probably too large for one response. Because this script is "
            "constrained to one API call, it will not retry or paginate; try "
            "lowering --max-results, adding --no-expand, or using --fields "
            "with only the fields you need."
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Jira returned {exc}: {resp.text}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        content_type = resp.headers.get("Content-Type", "unknown content type")
        raise RuntimeError(f"Jira did not return JSON ({content_type}).") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Jira returned a non-object JSON value.")
    issues = data.get("issues")
    if issues is None:
        data["issues"] = []
    elif not isinstance(issues, list):
        raise RuntimeError("Jira search response did not contain an issues array.")
    return data


def issue_filename(issue: dict[str, Any], fallback_index: int) -> str:
    key = issue.get("key") or issue.get("id") or f"issue-{fallback_index + 1}"
    return _safe_filename_stem(str(key), f"issue-{fallback_index + 1}") + ".json"


def write_issue_files(output_dir: str, issues: list[Any], compact: bool) -> int:
    os.makedirs(output_dir, exist_ok=True)
    written = 0
    used_names: dict[str, int] = {}

    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            print(f"[SKIP] issue #{index + 1}: non-object issue payload", file=sys.stderr)
            continue

        filename = issue_filename(issue, index)
        count = used_names.get(filename, 0) + 1
        used_names[filename] = count
        if count > 1:
            stem, ext = os.path.splitext(filename)
            filename = f"{stem}-{count}{ext}"

        path = os.path.abspath(os.path.join(output_dir, filename))
        with open(path, "w", encoding="utf-8") as fh:
            if compact:
                json.dump(issue, fh, separators=(",", ":"), ensure_ascii=False)
            else:
                json.dump(issue, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

        written += 1
        print(f"[ OK ] wrote {path}", file=sys.stderr)

    return written


def warn_if_truncated(data: dict[str, Any], issue_count: int, search_mode: str) -> None:
    """Warn when Jira says more issues exist but the one-call response ended."""
    if search_mode == "classic":
        total = data.get("total")
        if isinstance(total, int) and total > issue_count:
            print(
                f"[WARN] Jira reports {total} matching issue(s), but the one "
                f"response returned {issue_count}. Not paginating because the "
                "script is constrained to one API call.",
                file=sys.stderr,
            )
        return

    is_last = data.get("isLast")
    next_page_token = data.get("nextPageToken")
    if is_last is False or next_page_token:
        print(
            f"[WARN] Jira returned {issue_count} issue(s) and indicated another "
            "page exists. Not using nextPageToken because the script is "
            "constrained to one API call.",
            file=sys.stderr,
        )


def parse_expands(raw_expands: list[str]) -> tuple[str, ...]:
    return parse_csv_values(raw_expands)


def parse_fields(raw_fields: list[str]) -> tuple[str, ...]:
    return parse_csv_values(raw_fields)


def parse_csv_values(raw_values: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for raw in raw_values:
        for item in raw.split(","):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return tuple(values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Jira tickets from one project key or fix version "
        "using exactly one JQL search API call.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Project key or fix version. Uppercase Jira-looking values are "
        "treated as project keys; use --fix-version to force a version.",
    )
    parser.add_argument(
        "--project",
        metavar="KEY",
        help="Jira project key to index.",
    )
    parser.add_argument(
        "--fix-version",
        metavar="VERSION",
        help="Jira fix version to index across visible projects.",
    )
    parser.add_argument(
        "--base-url",
        help="Jira base URL. Falls back to $JIRA_BASE_URL, then $JIRA_URL.",
    )
    parser.add_argument(
        "--api-version",
        type=int,
        choices=(2, 3),
        default=DEFAULT_API_VERSION,
        help=f"Jira REST API version to use (default: {DEFAULT_API_VERSION}).",
    )
    parser.add_argument(
        "--search-mode",
        choices=("classic", "enhanced"),
        default=DEFAULT_SEARCH_MODE,
        help="Search endpoint style: classic uses /search; enhanced uses "
        f"/search/jql (default: {DEFAULT_SEARCH_MODE}).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Maximum issues to ask Jira to return in the one response. Jira "
        f"may cap this lower (default: {DEFAULT_MAX_RESULTS}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Directory for per-ticket JSON files (default: {DEFAULT_OUTPUT_DIR!r}).",
    )
    parser.add_argument(
        "--username",
        help="Jira Basic auth username/email. Falls back to $JIRA_USER, "
        "$JIRA_USERNAME, then $ATLASSIAN_EMAIL.",
    )
    parser.add_argument(
        "--password",
        help="Jira Basic auth password or API token. Falls back to "
        "$JIRA_API_TOKEN, $JIRA_TOKEN, $JIRA_PASSWORD, then $ATLASSIAN_API_TOKEN.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds for the one request (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--truststore",
        help="JKS truststore path for Jira TLS verification. Falls back to "
        "$JIRA_TRUSTSTORE, then $JIRA_TRUSTSTORE_PATH.",
    )
    parser.add_argument(
        "--truststore-password",
        help="Truststore password. Falls back to $JIRA_TRUSTSTORE_PASSWORD, "
        "then $TRUSTSTORE_PASSWORD.",
    )
    parser.add_argument(
        "--truststore-type",
        type=str.upper,
        default=DEFAULT_TRUSTSTORE_TYPE,
        choices=("JKS", "PKCS12"),
        help=f"Java truststore type (default: {DEFAULT_TRUSTSTORE_TYPE}).",
    )
    parser.add_argument(
        "--keytool",
        help="Path to keytool for JKS-to-PEM export. Falls back to $KEYTOOL, "
        "$JAVA_KEYTOOL, then 'keytool' on PATH.",
    )
    parser.add_argument(
        "--expand",
        action="append",
        default=[],
        help="Comma-separated Jira expand values. Defaults to a broad set of "
        "issue expansions. Repeatable. Passing this replaces the defaults.",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Do not send Jira expand values. Useful when the one-call payload "
        "is large enough to time out.",
    )
    parser.add_argument(
        "--fields",
        action="append",
        default=[],
        help="Comma-separated Jira fields to request. Defaults to '*all'. "
        "Repeatable. Example: --fields key,summary,status,fixVersions",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON instead of indented, human-readable JSON.",
    )
    parser.add_argument(
        "--print-jql",
        action="store_true",
        help="Print the generated JQL to stderr before fetching.",
    )
    parser.add_argument(
        "--print-request",
        action="store_true",
        help="Print the search URL and JSON request body to stderr before fetching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the request, then exit before truststore conversion "
        "or the Jira API call.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_results < 1:
        raise SystemExit("--max-results must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1 second.")
    if args.search_mode == "enhanced" and args.api_version != 3:
        raise SystemExit("--search-mode enhanced requires --api-version 3.")

    try:
        jql = build_jql(args.project, args.fix_version, args.target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    raw_base_url = args.base_url or _first_env(BASE_URL_ENV_VARS)
    if not raw_base_url:
        raise SystemExit(
            "No Jira base URL provided. Use --base-url or set JIRA_BASE_URL."
        )
    try:
        site = normalize_jira_site(raw_base_url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    username = args.username or _first_env(USERNAME_ENV_VARS)
    password = args.password or _first_env(PASSWORD_ENV_VARS)
    try:
        auth = build_auth(username, password)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    truststore_path = args.truststore or _first_env(TRUSTSTORE_ENV_VARS)
    truststore_password = (
        args.truststore_password or _first_env(TRUSTSTORE_PASSWORD_ENV_VARS)
    )
    keytool = args.keytool or _first_env(KEYTOOL_ENV_VARS) or "keytool"

    if args.print_jql:
        print(f"jql={jql}", file=sys.stderr)
    if auth is None:
        print("No Basic auth credentials provided.", file=sys.stderr)
    if truststore_path:
        print(
            f"Using {args.truststore_type} truststore for TLS verification: "
            f"{truststore_path}",
            file=sys.stderr,
        )

    fields = parse_fields(args.fields) if args.fields else DEFAULT_FIELDS
    expands = (
        ()
        if args.no_expand
        else parse_expands(args.expand)
        if args.expand
        else DEFAULT_EXPANDS
    )
    body = request_body(
        jql,
        max_results=args.max_results,
        search_mode=args.search_mode,
        fields=fields,
        expand=expands,
    )
    if args.print_request:
        print(
            "script="
            + os.path.abspath(__file__)
            + "\nurl="
            + search_url(site, args.api_version, args.search_mode)
            + "\nbody="
            + json.dumps(body, indent=2),
            file=sys.stderr,
        )
    if args.dry_run:
        if not args.print_request:
            print(
                json.dumps(
                    {
                        "script": os.path.abspath(__file__),
                        "url": search_url(site, args.api_version, args.search_mode),
                        "body": body,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
        return 0

    print(f"Fetching Jira issues from {site.base_url}...", file=sys.stderr)
    started = time.perf_counter()
    try:
        with tls_verify_bundle(
            truststore_path=truststore_path,
            truststore_password=truststore_password,
            truststore_type=args.truststore_type,
            keytool=keytool,
        ) as verify:
            data = fetch_issues_once(
                site,
                api_version=args.api_version,
                search_mode=args.search_mode,
                jql=jql,
                max_results=args.max_results,
                fields=fields,
                expand=expands,
                auth=auth,
                timeout=args.timeout,
                verify=verify,
            )
        issues = data["issues"]
        written = write_issue_files(args.output_dir, issues, compact=args.compact)
    except (OSError, requests.RequestException, RuntimeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    warn_if_truncated(data, len(issues), args.search_mode)
    elapsed = time.perf_counter() - started
    print(
        f"Done in {elapsed:.2f}s: wrote {written} ticket file(s) from one API call.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
