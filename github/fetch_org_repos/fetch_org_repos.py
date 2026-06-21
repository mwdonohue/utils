#!/usr/bin/env python3
"""Fetch metadata for every repository in a GitHub organization.

Spec
----
* Input      : the name of a GitHub organization.
* Output     : a JSON file containing metadata for all repos in the org.
* Constraint : exactly ONE API endpoint may be used. It may be called
               multiple times to walk through pagination.
* Goal       : minimize API calls first, then minimize wall-clock time.

The single endpoint used is:

    GET https://api.github.com/orgs/{org}/repos

Call-count minimization
-----------------------
The endpoint returns up to 100 repos per call (its hard cap), so the minimum
number of calls is ``ceil(repo_count / 100)`` and there is no way to do better
with this endpoint. We always request ``per_page=100``.

Performance (wall-clock) minimization
-------------------------------------
Naive pagination is *sequential*: you fetch page N, read its ``Link`` header to
discover page N+1, then fetch that, and so on -- one blocking round-trip per
page. Instead we exploit the ``rel="last"`` link GitHub returns on the FIRST
response, which reveals the total page count immediately. We then fetch the
first page, learn there are N pages, and request pages 2..N **concurrently**
with a bounded thread pool. This does not change the number of calls -- it just
collapses N serial round-trips into roughly one round-trip of wall time.

Other perf measures: HTTP keep-alive + a connection pool sized to the worker
count (via a mounted HTTPAdapter), automatic gzip (requests default), and
urllib3-level retries with backoff for transient 5xx / connection errors.

Authentication is optional but strongly recommended:
  * Without a token: only public repos are visible and the rate limit is
    60 requests/hour.
  * With a token: private repos the token can see are included and the rate
    limit is 5000 requests/hour.

The token is read (in order) from --token, $GITHUB_TOKEN, then $GH_TOKEN.

If GitHub Enterprise or a proxy uses an internal CA, a Java truststore can be
supplied for TLS verification:

    truststore path     : GITHUB_TRUSTSTORE, GITHUB_TRUSTSTORE_PATH
    truststore password : GITHUB_TRUSTSTORE_PASSWORD, TRUSTSTORE_PASSWORD
    keytool             : KEYTOOL, JAVA_KEYTOOL

Usage
-----
    python fetch_org_repos.py ORG_NAME
    python fetch_org_repos.py anthropics -o anthropics.json
    python fetch_org_repos.py big-org --workers 16
    GITHUB_TOKEN=ghp_xxx python fetch_org_repos.py my-private-org --type all
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
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ModuleNotFoundError:  # pragma: no cover - friendly message, not logic
    sys.exit(
        "This script needs the 'requests' library.\n"
        "Install it with:  python -m pip install requests"
    )

API_ROOT = "https://api.github.com"
# The one and only endpoint this tool is allowed to hit.
ENDPOINT_TEMPLATE = API_ROOT + "/orgs/{org}/repos"
MAX_PER_PAGE = 100  # GitHub's hard cap for this endpoint.
DEFAULT_WORKERS = 8
TRUSTSTORE_ENV_VARS = ("GITHUB_TRUSTSTORE", "GITHUB_TRUSTSTORE_PATH")
TRUSTSTORE_PASSWORD_ENV_VARS = ("GITHUB_TRUSTSTORE_PASSWORD", "TRUSTSTORE_PASSWORD")
KEYTOOL_ENV_VARS = ("KEYTOOL", "JAVA_KEYTOOL")
DEFAULT_TRUSTSTORE_TYPE = "JKS"
CERTIFICATE_RE = re.compile(
    r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.DOTALL,
)


def _first_env(names: tuple[str, ...]) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


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
            "set GITHUB_TRUSTSTORE_PASSWORD."
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

    fd, pem_path = tempfile.mkstemp(prefix="github-truststore-", suffix=".pem")
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


def build_session(token: str | None, pool_size: int) -> requests.Session:
    """Create a requests session with GitHub headers, keep-alive, and retries.

    ``pool_size`` sizes the underlying connection pool so concurrent page
    fetches can reuse pooled TLS connections instead of contending for a
    too-small pool. Transient 5xx / connection errors are retried with
    exponential backoff at the urllib3 layer; primary/secondary rate limits
    (403/429) are handled separately in ``_get`` because they require honoring
    GitHub's reset timestamp.
    """
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "fetch-org-repos-script",
        }
    )
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    return session


def _check_status(resp: requests.Response, org: str) -> None:
    """Translate non-success statuses into clear, actionable errors."""
    if resp.status_code == 404:
        raise SystemExit(
            f"Organization '{org}' not found (or no access). "
            "Check the name, and provide a token if it is private."
        )
    if resp.status_code == 401:
        raise SystemExit("Authentication failed: the provided token is invalid.")
    resp.raise_for_status()


def _sleep_for_rate_limit(resp: requests.Response) -> float | None:
    """Return seconds to wait if ``resp`` is a rate-limit rejection, else None."""
    if resp.status_code not in (403, 429):
        return None
    retry_after = resp.headers.get("Retry-After")
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")
    if retry_after is not None:
        return min(float(retry_after), 3600)
    if remaining == "0" and reset is not None:
        return min(max(0.0, float(reset) - time.time()) + 1.0, 3600)
    return None  # A 403 that is not a rate limit (e.g. bad token / no access).


def _get(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None,
    org: str,
    max_rate_limit_waits: int,
    verify: bool | str,
) -> requests.Response:
    """GET a URL, transparently waiting out rate limits, then validate status."""
    waits = 0
    while True:
        resp = session.get(url, params=params, timeout=30, verify=verify)
        sleep_for = _sleep_for_rate_limit(resp)
        if sleep_for is not None and waits < max_rate_limit_waits:
            print(
                f"  rate limited; sleeping {sleep_for:.0f}s before retrying...",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
            waits += 1
            continue
        _check_status(resp, org)
        return resp


def _last_page_number(resp: requests.Response) -> int:
    """Read the total page count from the first response's rel="last" link.

    Returns 1 when there is no ``last`` link (i.e. a single page of results).
    """
    last_url = resp.links.get("last", {}).get("url")
    if not last_url:
        return 1
    page_vals = parse_qs(urlparse(last_url).query).get("page")
    return int(page_vals[0]) if page_vals else 1


def fetch_all_repos(
    org: str,
    *,
    token: str | None = None,
    repo_type: str = "all",
    per_page: int = MAX_PER_PAGE,
    workers: int = DEFAULT_WORKERS,
    max_rate_limit_waits: int = 3,
    verify: bool | str = True,
) -> list[dict[str, Any]]:
    """Return metadata for every repo in ``org``, paging concurrently.

    Always uses the single ``/orgs/{org}/repos`` endpoint with ``per_page=100``
    so the call count stays at its ``ceil(n/100)`` floor; pages after the first
    are fetched in parallel to minimize wall-clock time.
    """
    per_page = min(per_page, MAX_PER_PAGE)
    workers = max(1, workers)
    session = build_session(token, pool_size=workers)
    base_url = ENDPOINT_TEMPLATE.format(org=org)
    base_params: dict[str, Any] = {
        "type": repo_type,
        "per_page": per_page,
        "sort": "full_name",
    }

    # First call doubles as discovery: it tells us how many pages exist.
    first = _get(session, base_url, base_params, org, max_rate_limit_waits, verify)
    last_page = _last_page_number(first)
    pages: dict[int, list[dict[str, Any]]] = {1: first.json()}
    print(
        f"  page 1: +{len(pages[1])} repos"
        + ("" if last_page == 1 else f"  ({last_page} pages total)"),
        file=sys.stderr,
    )

    if last_page > 1:
        def fetch_page(n: int) -> tuple[int, list[dict[str, Any]]]:
            params = dict(base_params, page=n)
            resp = _get(session, base_url, params, org, max_rate_limit_waits, verify)
            return n, resp.json()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_page, n) for n in range(2, last_page + 1)]
            for future in as_completed(futures):
                n, page = future.result()
                pages[n] = page
                print(f"  page {n}: +{len(page)} repos", file=sys.stderr)

    # Concatenate in page order so output is deterministic regardless of the
    # order in which concurrent requests completed.
    repos: list[dict[str, Any]] = []
    for n in range(1, last_page + 1):
        repos.extend(pages[n])
    return repos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch metadata for all repos in a GitHub organization "
        "and write it to a JSON file (uses only the /orgs/{org}/repos endpoint).",
    )
    parser.add_argument("org", help="GitHub organization name (e.g. 'anthropics').")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path. Defaults to '<org>-repos.json'.",
    )
    parser.add_argument(
        "--token",
        help="GitHub token. Falls back to $GITHUB_TOKEN, then $GH_TOKEN.",
    )
    parser.add_argument(
        "--truststore",
        help="JKS/PKCS12 truststore path for GitHub TLS verification. Falls "
        "back to $GITHUB_TRUSTSTORE, then $GITHUB_TRUSTSTORE_PATH.",
    )
    parser.add_argument(
        "--truststore-password",
        help="Truststore password. Falls back to $GITHUB_TRUSTSTORE_PASSWORD, "
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
        "--type",
        default="all",
        choices=["all", "public", "private", "forks", "sources", "member"],
        help="Which repos to include (default: all).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=MAX_PER_PAGE,
        help=f"Repos per request, max {MAX_PER_PAGE} (default: {MAX_PER_PAGE}). "
        "Leave at the max to minimize the number of API calls.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent page fetches (default: {DEFAULT_WORKERS}). "
        "Use 1 for fully sequential paging; lower it if you hit secondary "
        "rate limits.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON instead of indented, human-readable JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    output_path = args.output or f"{args.org}-repos.json"
    truststore_path = args.truststore or _first_env(TRUSTSTORE_ENV_VARS)
    truststore_password = (
        args.truststore_password or _first_env(TRUSTSTORE_PASSWORD_ENV_VARS)
    )
    keytool = args.keytool or _first_env(KEYTOOL_ENV_VARS) or "keytool"

    if not token:
        print(
            "No token provided -> only public repos, 60 requests/hour limit.",
            file=sys.stderr,
        )
    if truststore_path:
        print(
            f"Using {args.truststore_type} truststore for TLS verification: "
            f"{truststore_path}",
            file=sys.stderr,
        )

    print(f"Fetching repos for organization '{args.org}'...", file=sys.stderr)
    started = time.perf_counter()
    try:
        with tls_verify_bundle(
            truststore_path=truststore_path,
            truststore_password=truststore_password,
            truststore_type=args.truststore_type,
            keytool=keytool,
        ) as verify:
            repos = fetch_all_repos(
                args.org,
                token=token,
                repo_type=args.type,
                per_page=args.per_page,
                workers=args.workers,
                verify=verify,
            )
    except (OSError, requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    with open(output_path, "w", encoding="utf-8") as fh:
        if args.compact:
            json.dump(repos, fh, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(repos, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(
        f"Wrote {len(repos)} repos to {output_path} in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
