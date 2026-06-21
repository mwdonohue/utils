#!/usr/bin/env python3
"""Mirror-clone many Git repositories, fast.

Spec
----
* Input       : a list of repository URLs (CLI args, file(s), or stdin).
* Output      : the output of ``git clone --mirror <url>`` for each repo.
* Constraints : support a PAT; optimize for performance; give each repo its
                own directory; handle hundreds of repos in one run.

Performance
-----------
Mirror-cloning hundreds of repos is network-bound and embarrassingly parallel,
so the big win is running many ``git clone`` subprocesses at once. We use a
bounded thread pool (``--jobs``); each worker shells out to ``git`` (the GIL is
irrelevant because the work happens in the child process). We also:

* set ``GIT_TERMINAL_PROMPT=0`` so a single auth-less repo can never block the
  whole batch on an interactive username/password prompt;
* pass ``-c gc.auto=0`` to skip auto-gc churn during the clone;
* retry transient failures with exponential backoff, cleaning the partial
  directory between attempts;
* skip repos already mirrored (or ``--update`` them in place), so re-runs over a
  large list are cheap and idempotent.

Authentication (PAT)
--------------------
A token is read (in order) from --token, then $GIT_PAT, $GITHUB_TOKEN, $GH_TOKEN.
For ``https://`` remotes it is injected as an HTTP ``Authorization: Basic``
header via git's ``GIT_CONFIG_*`` environment variables, scoped to each repo's
host. This means the token is:

* never written to the mirror's on-disk config (unlike baking it into the
  remote URL), because env/`-c` config is not persisted to the clone;
* never visible in the process list (it's an env var, not an argv);
* never sent to a host other than the one being cloned.

URLs that already embed credentials are left untouched, and the token is
redacted from all printed output. SSH remotes (``git@host:...``) authenticate
with your SSH keys; the PAT does not apply to them.

If GitHub Enterprise or a proxy uses an internal CA, a Java truststore can be
supplied for HTTPS clone TLS verification:

    truststore path     : GITHUB_TRUSTSTORE, GITHUB_TRUSTSTORE_PATH
    truststore password : GITHUB_TRUSTSTORE_PASSWORD, TRUSTSTORE_PASSWORD
    keytool             : KEYTOOL, JAVA_KEYTOOL

Usage
-----
    python mirror_clone_repos.py https://github.com/owner/repo.git
    python mirror_clone_repos.py -i repos.txt -d ./mirrors -j 16
    python mirror_clone_repos.py -i anthropics-repos.json --prefer ssh
    GIT_PAT=ghp_xxx python mirror_clone_repos.py -i private-repos.txt --update
    python fetch_org_repos.py myorg -o repos.json && \
        python mirror_clone_repos.py -i repos.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urlsplit

DEFAULT_JOBS = 8
DEFAULT_RETRIES = 2
DEFAULT_DEST = "mirrors"
DEFAULT_TOKEN_USERNAME = "x-access-token"
TOKEN_ENV_VARS = ("GIT_PAT", "GITHUB_TOKEN", "GH_TOKEN")
TRUSTSTORE_ENV_VARS = ("GITHUB_TRUSTSTORE", "GITHUB_TRUSTSTORE_PATH")
TRUSTSTORE_PASSWORD_ENV_VARS = ("GITHUB_TRUSTSTORE_PASSWORD", "TRUSTSTORE_PASSWORD")
KEYTOOL_ENV_VARS = ("KEYTOOL", "JAVA_KEYTOOL")
DEFAULT_TRUSTSTORE_TYPE = "JKS"

# Keys to pull a clone URL from when the input is JSON (e.g. the output of the
# sibling fetch_org_repos.py). Order encodes the --prefer choice.
JSON_URL_KEYS = {
    "https": ("clone_url", "html_url", "git_url", "ssh_url"),
    "ssh": ("ssh_url", "clone_url", "git_url", "html_url"),
}

_print_lock = threading.Lock()
# Matches the "user[:pass]@" credential portion of any URL, for redaction.
_CRED_RE = re.compile(r"(\w+://)[^/@\s]+@")
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
def truststore_ca_file(
    truststore_path: str | None,
    truststore_password: str | None,
    truststore_type: str,
    keytool: str,
) -> Iterator[str | None]:
    """Yield a Git-compatible CA bundle path, converting JKS/PKCS12 to PEM."""
    if not truststore_path:
        yield None
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


# --------------------------------------------------------------------------- #
# URL parsing / destination layout
# --------------------------------------------------------------------------- #
@dataclass
class RepoUrl:
    """A parsed repository URL."""

    raw: str
    scheme: str  # "http", "https", "ssh", or "" for local/unknown
    host: str
    path: str  # owner/name (without a leading slash)
    has_userinfo: bool  # URL already embeds credentials


def parse_url(url: str) -> RepoUrl:
    """Parse an https/ssh/scp-style git URL into its parts."""
    url = url.strip()
    if "://" in url:
        parts = urlsplit(url)
        return RepoUrl(
            raw=url,
            scheme=parts.scheme,
            host=parts.hostname or "",
            path=parts.path.lstrip("/"),
            has_userinfo=bool(parts.username),
        )
    # scp-like SSH syntax: [user@]host:owner/repo.git
    if ":" in url and not os.path.isabs(url):
        userhost, _, path = url.partition(":")
        return RepoUrl(
            raw=url,
            scheme="ssh",
            host=userhost.split("@")[-1],
            path=path.lstrip("/"),
            has_userinfo="@" in userhost,
        )
    # Local path or something we don't recognize; clone it verbatim.
    return RepoUrl(raw=url, scheme="", host="", path=url, has_userinfo=False)


def _repo_segments(repo: RepoUrl) -> list[str]:
    """Path components without a trailing '.git', e.g. ['owner', 'repo']."""
    path = repo.path[:-4] if repo.path.endswith(".git") else repo.path
    segments = [seg for seg in path.split("/") if seg]
    return segments or ["repository"]


def dest_for(repo: RepoUrl, root: str, layout: str) -> str:
    """Compute the per-repo mirror directory, absolute.

    ``nested`` (default) reproduces ``<host>/<owner>/<repo>.git`` so repos with
    the same name from different owners or hosts never collide. ``flat`` puts
    everything directly under ``root`` as ``<owner>_<repo>.git``.
    """
    segments = _repo_segments(repo)
    if layout == "flat":
        name = "_".join(segments) + ".git"
        return os.path.abspath(os.path.join(root, name))
    parts = ([repo.host] if repo.host else []) + segments
    parts[-1] += ".git"
    return os.path.abspath(os.path.join(root, *parts))


def display_url(url: str) -> str:
    """Strip embedded credentials from a URL for safe display."""
    return _CRED_RE.sub(r"\1", url)


def redact(text: str, token: str | None) -> str:
    """Remove the token and any URL credentials from captured output."""
    if not text:
        return text
    if token:
        text = text.replace(token, "***")
    return _CRED_RE.sub(r"\1***@", text)


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def _urls_from_json_item(item: object, keys: tuple[str, ...]) -> list[str]:
    if isinstance(item, str):
        return [item]
    if isinstance(item, dict):
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                return [value]
    return []


def urls_from_json(data: object, keys: tuple[str, ...]) -> list[str]:
    """Extract clone URLs from parsed JSON (list/dict of repo objects)."""
    urls: list[str] = []
    if isinstance(data, list):
        for item in data:
            urls.extend(_urls_from_json_item(item, keys))
    elif isinstance(data, dict):
        if any(k in data for k in keys):  # a single repo object
            urls.extend(_urls_from_json_item(data, keys))
        else:  # a wrapper like {"repos": [...]}
            for value in data.values():
                if isinstance(value, list):
                    urls.extend(urls_from_json(value, keys))
    return urls


def parse_url_text(text: str, prefer: str) -> list[str]:
    """Parse a source's text as JSON (if it looks like JSON) or line-by-line."""
    stripped = text.lstrip()
    if stripped[:1] in "[{":
        try:
            return urls_from_json(json.loads(text), JSON_URL_KEYS[prefer])
        except json.JSONDecodeError:
            pass  # fall through and treat it as plain lines
    urls = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()  # strip comments and whitespace
        if line:
            urls.append(line)
    return urls


def load_urls(args: argparse.Namespace) -> list[str]:
    """Gather URLs from CLI args, --input files, and/or stdin; dedupe in order."""
    collected: list[str] = list(args.urls)

    sources = list(args.input)
    # Read stdin when explicitly asked ("-") or when nothing else was provided
    # and input is being piped in.
    if "-" in sources or (not collected and not sources and not sys.stdin.isatty()):
        collected += parse_url_text(sys.stdin.read(), args.prefer)
        sources = [s for s in sources if s != "-"]

    for path in sources:
        try:
            with open(path, encoding="utf-8") as fh:
                collected += parse_url_text(fh.read(), args.prefer)
        except OSError as exc:
            raise SystemExit(f"Cannot read input file '{path}': {exc}")

    seen: set[str] = set()
    unique: list[str] = []
    for url in collected:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


# --------------------------------------------------------------------------- #
# Cloning
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    url: str  # display (redacted) URL
    dest: str
    status: str  # cloned | updated | skipped | failed
    returncode: int = 0
    attempts: int = 0
    elapsed: float = 0.0
    output: str = ""


def is_bare_repo(path: str) -> bool:
    """True if ``path`` looks like a complete bare/mirror repository."""
    return all(
        os.path.exists(os.path.join(path, name))
        for name in ("HEAD", "objects", "refs")
    )


def build_env(
    repo: RepoUrl,
    token: str | None,
    username: str,
    ssl_ca_info: str | None,
) -> dict[str, str]:
    """Environment for the git child: no prompts, plus a scoped PAT header."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # never hang the batch on an auth prompt
    if ssl_ca_info:
        env["GIT_SSL_CAINFO"] = ssl_ca_info
    if token and repo.scheme in ("http", "https") and repo.host:
        creds = base64.b64encode(f"{username}:{token}".encode()).decode()
        header = f"Authorization: Basic {creds}"
        env["GIT_CONFIG_COUNT"] = "1"
        # Scope the header to this repo's host so the token never leaks to
        # another host in a mixed-host batch. Not persisted to the clone.
        env["GIT_CONFIG_KEY_0"] = f"http.{repo.scheme}://{repo.host}/.extraHeader"
        env["GIT_CONFIG_VALUE_0"] = header
    return env


def _run_git(cmd: list[str], env: dict[str, str], token: str | None) -> tuple[int, str]:
    """Run a git command, returning (exit code, redacted combined output)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except FileNotFoundError:
        raise SystemExit("'git' was not found on PATH. Install Git and retry.")
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, redact(output, token)


def clone_one(repo: RepoUrl, args: argparse.Namespace, token: str | None) -> Result:
    """Mirror-clone (or update/skip) a single repository."""
    dest = dest_for(repo, args.dest, args.layout)
    shown = display_url(repo.raw)
    start = time.perf_counter()

    use_token = token if not repo.has_userinfo else None
    if repo.has_userinfo and token:
        # URL already carries credentials; respect them rather than overriding.
        pass
    env = build_env(repo, use_token, args.token_username, args.ssl_ca_info)

    already = os.path.isdir(dest) and is_bare_repo(dest)
    if already and not args.force:
        if args.update:
            code, out = _run_git(
                ["git", "--git-dir", dest, "remote", "update", "--prune"],
                env,
                use_token,
            )
            status = "updated" if code == 0 else "failed"
            return Result(shown, dest, status, code, 1, time.perf_counter() - start, out)
        return Result(shown, dest, "skipped", 0, 0, time.perf_counter() - start,
                      "already mirrored; use --update to refresh or --force to re-clone")

    cmd = ["git", "-c", "gc.auto=0"]
    if use_token and repo.scheme in ("http", "https"):
        # We supply auth via the header; disable helpers so a missing/expired
        # cached credential can't trigger a GUI prompt mid-batch.
        cmd += ["-c", "credential.helper="]
    cmd += ["clone", "--mirror", repo.raw, dest]

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    outputs: list[str] = []
    code = -1
    for attempt in range(1, args.retries + 2):
        # Clear any leftover partial directory from a prior failed attempt.
        if os.path.exists(dest) and not is_bare_repo(dest):
            shutil.rmtree(dest, ignore_errors=True)
        code, out = _run_git(cmd, env, use_token)
        outputs.append(out)
        if code == 0:
            return Result(shown, dest, "cloned", 0, attempt,
                          time.perf_counter() - start, out)
        if attempt <= args.retries:
            backoff = min(2 ** (attempt - 1), 30)
            outputs.append(f"[retrying in {backoff}s after exit {code}]\n")
            time.sleep(backoff)

    shutil.rmtree(dest, ignore_errors=True)  # don't leave a broken mirror behind
    return Result(shown, dest, "failed", code, args.retries + 1,
                  time.perf_counter() - start, "".join(outputs))


def emit(result: Result) -> None:
    """Print one repo's clone output (stdout) and a status line (stderr)."""
    badge = {"cloned": "OK", "updated": "UPD", "skipped": "SKIP", "failed": "FAIL"}
    with _print_lock:
        print(f"\n=== {result.status.upper()}: {result.url} -> {result.dest} ===")
        if result.output.strip():
            print(result.output.rstrip())
        sys.stdout.flush()
        print(
            f"[{badge.get(result.status, '?'):>4}] {result.url} "
            f"({result.elapsed:.1f}s)",
            file=sys.stderr,
        )
        sys.stderr.flush()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror-clone many Git repositories in parallel "
        "(git clone --mirror), with PAT support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("urls", nargs="*", help="Repository URLs to mirror.")
    parser.add_argument(
        "-i", "--input", action="append", default=[], metavar="FILE",
        help="Read URLs from FILE (one per line, or the JSON output of "
        "fetch_org_repos.py). Repeatable. Use '-' for stdin.",
    )
    parser.add_argument(
        "-d", "--dest", default=DEFAULT_DEST, metavar="DIR",
        help=f"Root directory for the mirrors (default: '{DEFAULT_DEST}').",
    )
    parser.add_argument(
        "--layout", choices=("nested", "flat"), default="nested",
        help="nested: <dest>/<host>/<owner>/<repo>.git (collision-safe, "
        "default). flat: <dest>/<owner>_<repo>.git.",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=DEFAULT_JOBS, metavar="N",
        help=f"Concurrent clones (default: {DEFAULT_JOBS}). Raise for more "
        "throughput; lower if you hit server-side rate limits.",
    )
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_RETRIES, metavar="N",
        help=f"Retries per repo on failure, with backoff (default: "
        f"{DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--token", help="PAT for https remotes. Falls back to "
        + ", ".join(f"${v}" for v in TOKEN_ENV_VARS) + ".",
    )
    parser.add_argument(
        "--token-username", default=DEFAULT_TOKEN_USERNAME, metavar="USER",
        help="Username paired with the PAT in the Basic auth header "
        f"(default: '{DEFAULT_TOKEN_USERNAME}'; works for GitHub).",
    )
    parser.add_argument(
        "--truststore",
        help="JKS/PKCS12 truststore path for HTTPS clone TLS verification. "
        "Falls back to $GITHUB_TRUSTSTORE, then $GITHUB_TRUSTSTORE_PATH.",
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
        "--prefer", choices=("https", "ssh"), default="https",
        help="Which URL field to pick from JSON input (default: https).",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="If a mirror already exists, refresh it (remote update --prune) "
        "instead of skipping.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete and re-clone mirrors that already exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be cloned and where, without running git.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.jobs = max(1, args.jobs)
    args.retries = max(0, args.retries)

    token = args.token or next(
        (os.environ[v] for v in TOKEN_ENV_VARS if os.environ.get(v)), None
    )
    truststore_path = args.truststore or _first_env(TRUSTSTORE_ENV_VARS)
    truststore_password = (
        args.truststore_password or _first_env(TRUSTSTORE_PASSWORD_ENV_VARS)
    )
    keytool = args.keytool or _first_env(KEYTOOL_ENV_VARS) or "keytool"

    urls = load_urls(args)
    if not urls:
        raise SystemExit(
            "No repository URLs provided. Pass URLs as arguments, via "
            "-i/--input FILE, or on stdin. See --help."
        )

    repos = [parse_url(u) for u in urls]

    if args.dry_run:
        for repo in repos:
            print(f"{display_url(repo.raw)}  ->  {dest_for(repo, args.dest, args.layout)}")
        print(f"\n{len(repos)} repo(s) would be mirrored into "
              f"'{os.path.abspath(args.dest)}'.", file=sys.stderr)
        return 0

    if not token:
        print("No PAT provided -> private https repos will fail; public repos "
              "and SSH remotes are unaffected.", file=sys.stderr)
    if truststore_path:
        print(
            f"Using {args.truststore_type} truststore for HTTPS clone TLS "
            f"verification: {truststore_path}",
            file=sys.stderr,
        )
    print(f"Mirroring {len(repos)} repo(s) into '{os.path.abspath(args.dest)}' "
          f"with {args.jobs} parallel job(s)...", file=sys.stderr)

    started = time.perf_counter()
    results: list[Result] = []
    try:
        with truststore_ca_file(
            truststore_path=truststore_path,
            truststore_password=truststore_password,
            truststore_type=args.truststore_type,
            keytool=keytool,
        ) as ssl_ca_info:
            args.ssl_ca_info = ssl_ca_info
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {
                    pool.submit(clone_one, repo, args, token): repo for repo in repos
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    emit(result)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = ", ".join(f"{counts[s]} {s}" for s in
                        ("cloned", "updated", "skipped", "failed") if s in counts)
    print(f"\nDone in {elapsed:.1f}s: {summary or '0 done'}", file=sys.stderr)

    failures = [r for r in results if r.status == "failed"]
    if failures:
        print("Failed:", file=sys.stderr)
        for r in failures:
            tail = r.output.strip().splitlines()[-1:] or ["(no output)"]
            print(f"  - {r.url} (exit {r.returncode}): {tail[0]}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
