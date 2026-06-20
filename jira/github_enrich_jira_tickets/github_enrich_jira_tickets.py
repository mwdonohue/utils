#!/usr/bin/env python3
"""Fetch Git Integration for Jira data for a list of Jira issue IDs.

Spec
----
* Input  : a list of Jira issue IDs/keys.
* Output : one JSON document containing raw output from Git Integration for
           Jira's ``{jira_url}/rest/gitplugin/1.0`` API.

By default, the script fetches commits for each issue using:

    GET {jira}/rest/gitplugin/1.0/issues/{issueKey}/commits

That is the issue-scoped Git Integration for Jira endpoint documented for
commit enrichment. Optional resources can be included with ``--resource``:

    commits   GET /issues/{issueKey}/commits
    branches  GET /issues/branches?key={issueKey}
    details   GET /issuegitdetails/issue/{issueKey}
    tags      GET /issuegitdetails/issue/{issueKey}/tag

Authentication is optional. Credentials are read from CLI args first, then
environment variables:

    base URL     : JIRA_BASE_URL, JIRA_URL
    username     : JIRA_USER, JIRA_USERNAME, ATLASSIAN_EMAIL
    password     : JIRA_API_TOKEN, JIRA_TOKEN, JIRA_PASSWORD,
                   ATLASSIAN_API_TOKEN
    bearer token : JIRA_BEARER_TOKEN
    truststore   : JIRA_TRUSTSTORE, JIRA_TRUSTSTORE_PATH
    trust pass   : JIRA_TRUSTSTORE_PASSWORD, TRUSTSTORE_PASSWORD
    keytool      : KEYTOOL, JAVA_KEYTOOL

Usage
-----
    python github_enrich_jira_tickets.py --base-url https://jira.example.com ABC-1 ABC-2
    python github_enrich_jira_tickets.py -i issue-ids.json -o git-enrichment.json
    python github_enrich_jira_tickets.py -i issue-ids.txt --resource all --show-files
    JIRA_BASE_URL=https://jira.example.com JIRA_USER=me JIRA_API_TOKEN=xxx \
      python github_enrich_jira_tickets.py ABC-1 ABC-2
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from urllib.parse import quote, urlsplit, urlunsplit

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

BASE_URL_ENV_VARS = ("JIRA_BASE_URL", "JIRA_URL")
USERNAME_ENV_VARS = ("JIRA_USER", "JIRA_USERNAME", "ATLASSIAN_EMAIL")
PASSWORD_ENV_VARS = (
    "JIRA_API_TOKEN",
    "JIRA_TOKEN",
    "JIRA_PASSWORD",
    "ATLASSIAN_API_TOKEN",
)
BEARER_TOKEN_ENV_VARS = ("JIRA_BEARER_TOKEN",)
TRUSTSTORE_ENV_VARS = ("JIRA_TRUSTSTORE", "JIRA_TRUSTSTORE_PATH")
TRUSTSTORE_PASSWORD_ENV_VARS = ("JIRA_TRUSTSTORE_PASSWORD", "TRUSTSTORE_PASSWORD")
KEYTOOL_ENV_VARS = ("KEYTOOL", "JAVA_KEYTOOL")
DEFAULT_TIMEOUT = 30
DEFAULT_WORKERS = 8
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 0.5
DEFAULT_TRUSTSTORE_TYPE = "JKS"
DEFAULT_RESOURCES = ("commits",)
ALL_RESOURCES = ("commits", "branches", "details", "tags")
ISSUE_ID_SPLIT_RE = re.compile(r"[\s,]+")
CERTIFICATE_RE = re.compile(
    r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.DOTALL,
)


@dataclass(frozen=True)
class JiraSite:
    """Normalized Jira site details."""

    raw: str
    base_url: str


@dataclass(frozen=True)
class ResourceRequest:
    """One Git Integration for Jira API request to make."""

    issue_id: str
    resource: str
    url: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ResourceResult:
    """Raw payload fetched for one issue/resource pair."""

    issue_id: str
    resource: str
    url: str
    payload: Any


@dataclass(frozen=True)
class ResourceError(Exception):
    """Serializable error for one issue/resource pair."""

    issue_id: str
    resource: str
    url: str
    message: str
    status_code: int | None = None
    response_text: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "url": self.url,
            "message": self.message,
        }
        if self.status_code is not None:
            data["statusCode"] = self.status_code
        if self.response_text:
            data["responseText"] = self.response_text
        return data


def _first_env(names: tuple[str, ...]) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def normalize_jira_site(raw_url: str) -> JiraSite:
    """Normalize a Jira URL into the base URL above ``/rest/gitplugin/1.0``."""
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
            "--password, --bearer-token, or Jira env vars instead"
        )
    if parts.query or parts.fragment:
        raise ValueError("Jira base URL must not include a query string or fragment")

    path = parts.path.rstrip("/")
    for marker in ("/rest/gitplugin/1.0", "/rest/api/"):
        index = path.find(marker)
        if index >= 0:
            path = path[:index].rstrip("/")
            break

    base_url = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return JiraSite(raw=raw_url, base_url=base_url)


def git_plugin_base_url(site: JiraSite) -> str:
    parts = urlsplit(site.base_url)
    path = parts.path.rstrip("/") + "/rest/gitplugin/1.0"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def git_plugin_url(site: JiraSite, path: str) -> str:
    if not path.startswith("/"):
        raise ValueError("Git Integration path must start with '/'")
    return git_plugin_base_url(site) + path


def request_for(
    site: JiraSite,
    issue_id: str,
    resource: str,
    *,
    show_files: bool,
) -> ResourceRequest:
    """Build a Git Integration for Jira request for an issue/resource pair."""
    quoted_issue = quote(issue_id, safe="")

    if resource == "commits":
        params: dict[str, Any] = {}
        if show_files:
            params["showFiles"] = "true"
        return ResourceRequest(
            issue_id=issue_id,
            resource=resource,
            url=git_plugin_url(site, f"/issues/{quoted_issue}/commits"),
            params=params,
        )
    if resource == "branches":
        return ResourceRequest(
            issue_id=issue_id,
            resource=resource,
            url=git_plugin_url(site, "/issues/branches"),
            params={"key": issue_id},
        )
    if resource == "details":
        return ResourceRequest(
            issue_id=issue_id,
            resource=resource,
            url=git_plugin_url(site, f"/issuegitdetails/issue/{quoted_issue}"),
            params={},
        )
    if resource == "tags":
        return ResourceRequest(
            issue_id=issue_id,
            resource=resource,
            url=git_plugin_url(site, f"/issuegitdetails/issue/{quoted_issue}/tag"),
            params={},
        )

    raise ValueError(f"unsupported resource: {resource}")


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def issue_ids_from_json_value(value: Any) -> list[str]:
    """Extract issue IDs from supported JSON input shapes."""
    if isinstance(value, list):
        issue_ids: list[str] = []
        for item in value:
            if isinstance(item, (str, int)):
                issue_ids.append(str(item))
            elif isinstance(item, dict):
                for key in ("issueId", "issue_id", "issueKey", "key", "id"):
                    if key in item and item[key] is not None:
                        issue_ids.append(str(item[key]))
                        break
                else:
                    raise ValueError(
                        "issue objects must contain one of: issueId, issue_id, "
                        "issueKey, key, id"
                    )
            else:
                raise ValueError(
                    "JSON issue lists must contain strings, numbers, or issue objects"
                )
        return issue_ids

    if isinstance(value, dict):
        for key in ("issueIds", "issue_ids", "issueKeys", "issue_keys", "issues", "keys"):
            if key in value:
                return issue_ids_from_json_value(value[key])
        raise ValueError(
            "JSON object input must contain one of: issueIds, issue_ids, "
            "issueKeys, issue_keys, issues, keys"
        )

    if isinstance(value, (str, int)):
        return [str(value)]

    raise ValueError("JSON input must be an array, object, string, or number")


def parse_issue_id_text(text: str, label: str) -> list[str]:
    """Parse issue IDs from JSON, newline-delimited text, or comma-delimited text."""
    text = text.strip()
    if not text:
        return []

    if text[0] in "[{\"" or text[0].isdigit():
        try:
            return issue_ids_from_json_value(json.loads(text))
        except json.JSONDecodeError:
            pass
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc

    return [item for item in ISSUE_ID_SPLIT_RE.split(text) if item]


def load_issue_ids(input_path: str | None, positional_issue_ids: list[str]) -> list[str]:
    """Load issue IDs from ``--input``, positional args, or piped stdin."""
    raw_ids: list[str] = []

    if input_path:
        try:
            if input_path == "-":
                raw_ids.extend(parse_issue_id_text(sys.stdin.read(), "stdin"))
            else:
                with open(input_path, encoding="utf-8") as fh:
                    raw_ids.extend(parse_issue_id_text(fh.read(), input_path))
        except OSError as exc:
            raise SystemExit(f"Cannot read input file '{input_path}': {exc}") from exc
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    raw_ids.extend(positional_issue_ids)

    if not input_path and not positional_issue_ids and not sys.stdin.isatty():
        try:
            raw_ids.extend(parse_issue_id_text(sys.stdin.read(), "stdin"))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    issue_ids = dedupe_preserving_order(raw_ids)
    if not issue_ids:
        raise SystemExit(
            "No issue IDs provided. Pass issue IDs as arguments, use --input, "
            "or pipe them on stdin."
        )
    return issue_ids


def parse_resources(raw_resources: list[str]) -> tuple[str, ...]:
    if not raw_resources:
        return DEFAULT_RESOURCES

    values: list[str] = []
    for raw in raw_resources:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if item == "all":
                for resource in ALL_RESOURCES:
                    if resource not in values:
                        values.append(resource)
                continue
            if item not in ALL_RESOURCES:
                raise SystemExit(
                    f"Unknown resource {item!r}. Choose from: "
                    + ", ".join((*ALL_RESOURCES, "all"))
                )
            if item not in values:
                values.append(item)
    return tuple(values) if values else DEFAULT_RESOURCES


def build_auth(
    username: str | None,
    password: str | None,
    bearer_token: str | None,
) -> HTTPBasicAuth | None:
    if bearer_token and (username or password):
        raise ValueError(
            "use either --bearer-token or Basic auth username/password, not both"
        )
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
    """Export a Java truststore's certificates into a temporary PEM bundle."""
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
    """Yield the requests ``verify`` value, converting JKS/PKCS12 to PEM."""
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


def build_session(
    bearer_token: str | None,
    pool_size: int,
    retries: int,
    retry_backoff: float,
) -> requests.Session:
    """Create a requests session with JSON headers, keep-alive, and GET retries."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "github-enrich-jira-tickets-script",
        }
    )
    if bearer_token:
        session.headers["Authorization"] = f"Bearer {bearer_token}"

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


def fetch_resource(
    session: requests.Session,
    request: ResourceRequest,
    auth: HTTPBasicAuth | None,
    timeout: int,
    verify: bool | str,
) -> ResourceResult:
    """Fetch one Git Integration for Jira resource and return its raw JSON."""
    resp = session.get(
        request.url,
        params=request.params,
        auth=auth,
        timeout=timeout,
        verify=verify,
        allow_redirects=False,
    )

    if 300 <= resp.status_code < 400:
        location = resp.headers.get("Location", "(missing Location header)")
        raise ResourceError(
            issue_id=request.issue_id,
            resource=request.resource,
            url=resp.url,
            status_code=resp.status_code,
            message=(
                f"{request.url} redirected to {location}. Provide the final "
                "Jira URL so request output is stable."
            ),
        )
    if resp.status_code == 401:
        raise ResourceError(
            issue_id=request.issue_id,
            resource=request.resource,
            url=resp.url,
            status_code=resp.status_code,
            message="Jira rejected the credentials (HTTP 401).",
        )
    if resp.status_code == 403:
        raise ResourceError(
            issue_id=request.issue_id,
            resource=request.resource,
            url=resp.url,
            status_code=resp.status_code,
            message=(
                "Jira denied access (HTTP 403). Check Browse Projects and "
                "View Development Tools permissions for the supplied user/token."
            ),
        )
    if resp.status_code == 404:
        raise ResourceError(
            issue_id=request.issue_id,
            resource=request.resource,
            url=resp.url,
            status_code=resp.status_code,
            message=(
                "Git Integration for Jira endpoint was not found (HTTP 404). "
                "Check the Jira URL, app installation, and issue ID/key."
            ),
            response_text=resp.text,
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise ResourceError(
            issue_id=request.issue_id,
            resource=request.resource,
            url=resp.url,
            status_code=resp.status_code,
            message=f"Jira returned {exc}.",
            response_text=resp.text,
        ) from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        content_type = resp.headers.get("Content-Type", "unknown content type")
        raise ResourceError(
            issue_id=request.issue_id,
            resource=request.resource,
            url=resp.url,
            status_code=resp.status_code,
            message=f"Jira did not return JSON ({content_type}).",
            response_text=resp.text[:2000],
        ) from exc

    return ResourceResult(
        issue_id=request.issue_id,
        resource=request.resource,
        url=resp.url,
        payload=payload,
    )


def fetch_all_resources(
    site: JiraSite,
    issue_ids: list[str],
    resources: tuple[str, ...],
    *,
    show_files: bool,
    auth: HTTPBasicAuth | None,
    bearer_token: str | None,
    timeout: int,
    verify: bool | str,
    workers: int,
    retries: int,
    retry_backoff: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Fetch requested resources concurrently and return payloads plus errors."""
    requests_to_make = [
        request_for(site, issue_id, resource, show_files=show_files)
        for issue_id in issue_ids
        for resource in resources
    ]
    pool_size = max(1, min(workers, len(requests_to_make)))
    results: dict[str, dict[str, Any]] = {issue_id: {} for issue_id in issue_ids}
    errors: dict[str, dict[str, dict[str, Any]]] = {
        issue_id: {} for issue_id in issue_ids
    }

    session = build_session(
        bearer_token=bearer_token,
        pool_size=pool_size,
        retries=retries,
        retry_backoff=retry_backoff,
    )
    try:
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            future_to_request = {
                pool.submit(
                    fetch_resource,
                    session,
                    request,
                    auth,
                    timeout,
                    verify,
                ): request
                for request in requests_to_make
            }
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                except ResourceError as exc:
                    errors[exc.issue_id][exc.resource] = exc.to_json()
                    print(
                        f"[FAIL] {exc.issue_id} :: {exc.resource}: {exc.message}",
                        file=sys.stderr,
                    )
                except requests.RequestException as exc:
                    errors[request.issue_id][request.resource] = {
                        "url": request.url,
                        "message": str(exc),
                    }
                    print(
                        f"[FAIL] {request.issue_id} :: {request.resource}: {exc}",
                        file=sys.stderr,
                    )
                else:
                    results[result.issue_id][result.resource] = result.payload
                    print(
                        f"[ OK ] {result.issue_id} :: {result.resource}",
                        file=sys.stderr,
                    )
    finally:
        session.close()

    results = {
        issue_id: {
            resource: results[issue_id][resource]
            for resource in resources
            if resource in results[issue_id]
        }
        for issue_id in issue_ids
    }
    errors = {
        issue_id: {
            resource: errors[issue_id][resource]
            for resource in resources
            if resource in errors[issue_id]
        }
        for issue_id in issue_ids
        if errors[issue_id]
    }
    return results, errors


def build_output(
    site: JiraSite,
    issue_ids: list[str],
    resources: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    errors: dict[str, dict[str, dict[str, Any]]],
    elapsed: float,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for issue_id in issue_ids:
        issue: dict[str, Any] = {
            "issueId": issue_id,
            "gitPlugin": results.get(issue_id, {}),
        }
        if issue_id in errors:
            issue["errors"] = errors[issue_id]
        issues.append(issue)

    return {
        "jiraBaseUrl": site.base_url,
        "gitPluginApi": git_plugin_base_url(site),
        "resources": list(resources),
        "issueCount": len(issue_ids),
        "elapsedSeconds": round(elapsed, 3),
        "issues": issues,
    }


def write_json_output(path: str | None, data: Any, compact: bool) -> None:
    kwargs = (
        {"separators": (",", ":")}
        if compact
        else {"indent": 2}
    )
    text = json.dumps(data, ensure_ascii=False, **kwargs) + "\n"
    if path and path != "-":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch raw Git Integration for Jira data for Jira issue IDs.",
    )
    parser.add_argument(
        "issue_ids",
        nargs="*",
        help="Jira issue IDs/keys such as ABC-123. Can be mixed with --input.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="File containing issue IDs as a JSON array/object, newline list, "
        "or comma-delimited text. Use '-' for stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path. Defaults to stdout. Use '-' for stdout.",
    )
    parser.add_argument(
        "--base-url",
        help="Jira base URL. Falls back to $JIRA_BASE_URL, then $JIRA_URL.",
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
        "--bearer-token",
        help="Jira bearer token. Falls back to $JIRA_BEARER_TOKEN. Cannot be "
        "combined with Basic auth.",
    )
    parser.add_argument(
        "--truststore",
        help="JKS/PKCS12 truststore path for Jira TLS verification. Falls back "
        "to $JIRA_TRUSTSTORE, then $JIRA_TRUSTSTORE_PATH.",
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
        help="Path to keytool for truststore-to-PEM export. Falls back to "
        "$KEYTOOL, $JAVA_KEYTOOL, then 'keytool' on PATH.",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="NAME",
        help="Git Integration resource to fetch: commits, branches, details, "
        "tags, or all. Defaults to commits. Repeatable and comma-delimited.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="For commits, pass showFiles=true so changed files are included "
        "when the Jira app supports it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent Jira requests (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds for each request (default: {DEFAULT_TIMEOUT}).",
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
        "--compact",
        action="store_true",
        help="Write minified JSON instead of indented, human-readable JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request plan as JSON and exit before calling Jira.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1 second.")
    if args.retries < 0:
        raise SystemExit("--retries must be 0 or greater.")
    if args.retry_backoff < 0:
        raise SystemExit("--retry-backoff must be 0 or greater.")

    raw_base_url = args.base_url or _first_env(BASE_URL_ENV_VARS)
    if not raw_base_url:
        raise SystemExit(
            "No Jira base URL provided. Use --base-url or set JIRA_BASE_URL."
        )
    try:
        site = normalize_jira_site(raw_base_url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    resources = parse_resources(args.resource)
    issue_ids = load_issue_ids(args.input, args.issue_ids)

    username = args.username or _first_env(USERNAME_ENV_VARS)
    password = args.password or _first_env(PASSWORD_ENV_VARS)
    bearer_token = args.bearer_token or _first_env(BEARER_TOKEN_ENV_VARS)
    try:
        auth = build_auth(username, password, bearer_token)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    truststore_path = args.truststore or _first_env(TRUSTSTORE_ENV_VARS)
    truststore_password = (
        args.truststore_password or _first_env(TRUSTSTORE_PASSWORD_ENV_VARS)
    )
    keytool = args.keytool or _first_env(KEYTOOL_ENV_VARS) or "keytool"

    if args.dry_run:
        requests_to_make = [
            request_for(site, issue_id, resource, show_files=args.show_files)
            for issue_id in issue_ids
            for resource in resources
        ]
        write_json_output(
            args.output,
            {
                "jiraBaseUrl": site.base_url,
                "gitPluginApi": git_plugin_base_url(site),
                "resources": list(resources),
                "issueIds": issue_ids,
                "tls": {
                    "verify": True,
                    "truststore": truststore_path,
                    "truststoreType": args.truststore_type,
                    "keytool": keytool,
                },
                "requests": [
                    {
                        "issueId": request.issue_id,
                        "resource": request.resource,
                        "method": "GET",
                        "url": request.url,
                        "params": request.params,
                    }
                    for request in requests_to_make
                ],
            },
            compact=args.compact,
        )
        return 0

    if auth is None and not bearer_token:
        print("No Jira credentials provided.", file=sys.stderr)
    if truststore_path:
        print(
            f"Using {args.truststore_type} truststore for TLS verification: "
            f"{truststore_path}",
            file=sys.stderr,
        )

    print(
        f"Fetching {len(resources)} Git Integration resource(s) for "
        f"{len(issue_ids)} issue(s) from {site.base_url}...",
        file=sys.stderr,
    )
    started = time.perf_counter()
    try:
        with tls_verify_bundle(
            truststore_path=truststore_path,
            truststore_password=truststore_password,
            truststore_type=args.truststore_type,
            keytool=keytool,
        ) as verify:
            results, errors = fetch_all_resources(
                site=site,
                issue_ids=issue_ids,
                resources=resources,
                show_files=args.show_files,
                auth=auth,
                bearer_token=bearer_token,
                timeout=args.timeout,
                verify=verify,
                workers=args.workers,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    output = build_output(
        site=site,
        issue_ids=issue_ids,
        resources=resources,
        results=results,
        errors=errors,
        elapsed=elapsed,
    )
    try:
        write_json_output(args.output, output, compact=args.compact)
    except OSError as exc:
        print(f"[FAIL] cannot write output JSON: {exc}", file=sys.stderr)
        return 1

    failure_count = sum(len(resource_errors) for resource_errors in errors.values())
    print(
        f"Done in {elapsed:.2f}s: "
        f"{len(issue_ids) * len(resources) - failure_count} succeeded, "
        f"{failure_count} failed.",
        file=sys.stderr,
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
