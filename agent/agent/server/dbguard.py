"""HTTP source admission and deployment-specific outbound URL policies.

SQL sources are explicit optional operations. Files enter through workspace handles;
optional host and scheme restrictions continue to apply to other connectors.
HTTP fetch clients share the host checks through check_outbound_url.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from fastapi import HTTPException

# Schemes that read a LOCAL FILE rather than talk to a server, blocked by the
# REXGRAPH_DB_SAFE preset because they can exfiltrate arbitrary files.
_FILE_SCHEMES = {"local", "file", "rex", "rcbd", "rcbf", "csv"}


def _csv_env(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _policy_active() -> bool:
    return bool(
        os.environ.get("REXGRAPH_DB_SAFE") == "1"
        or os.environ.get("REXGRAPH_DB_BLOCK_LOCAL") == "1"
        or _csv_env("REXGRAPH_ALLOWED_DB_SCHEMES")
        or _csv_env("REXGRAPH_ALLOWED_DB_HOSTS")
    )


def _ip_blocked(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (ip.is_loopback or ip.is_private or ip.is_link_local
            or ip.is_reserved or ip.is_unspecified or ip.is_multicast)


def _host_is_local_or_private(host: str) -> bool:
    if not host:
        return False
    if host.lower() in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    # Literal IP -> check directly.
    try:
        ipaddress.ip_address(host)
        return _ip_blocked(host)
    except ValueError:
        pass
    # Hostname -> resolve and block if ANY resolved address is loopback/private.
    # This closes the hostname->private IP SSRF case (e.g. a name that resolves to
    # 169.254.169.254). Unresolvable names fall through: the real connection just
    # fails. Full DNS rebinding defense needs connect time IP pinning (out of scope).
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return any(_ip_blocked(info[4][0]) for info in infos)


def check_outbound_url(url: str) -> None:
    """Raise HTTPException(400) if `url` names something this deployment will not fetch.

    Same SSRF question as `check_db_uri`, different policy question, so it shares the
    host test and not the scheme list: a TrustGraph read travels over http, which no
    REXGRAPH_ALLOWED_DB_SCHEMES would ever name, and running it through the database
    policy would reject every legitimate url the moment that policy was turned on.

    The scheme check is unconditional because it is not policy: `file://` handed to an
    HTTP client is a local file read, and no deployment wants that from a request body.
    The host policy follows the same opt in the database one does, so local and single
    operator use is unchanged, which also means a default deployment has no host check.

    A REDIRECT is not covered. This sees the url the caller supplied, and the client that
    fetches it follows redirects on its own, so a permitted host can forward to one that
    is not. Closing that means refusing redirects at the client, which belongs with the
    client rather than here.
    """
    if not url:
        return
    if "://" not in url:
        # check_db_uri returns early here because a bare name like "edgelist" is a valid
        # in memory scheme. Nothing of the sort is a fetchable URL, so the same early
        # return let "//169.254.169.254/latest/" past without any check at all.
        raise HTTPException(400, "not a fetchable url")
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise HTTPException(400, f"'{scheme}://' is not a fetchable scheme")
    host = parsed.hostname or ""
    if (os.environ.get("REXGRAPH_DB_SAFE") == "1"
            or os.environ.get("REXGRAPH_BLOCK_LOCAL_FETCH") == "1") and \
            _host_is_local_or_private(host):
        raise HTTPException(
            400, f"fetch from loopback/private host '{host}' is blocked (anti-SSRF)")
    allowed = _csv_env("REXGRAPH_ALLOWED_FETCH_HOSTS")
    if allowed is not None and host.lower() not in allowed:
        raise HTTPException(
            400, f"host '{host or '(none)'}' is not permitted "
                 f"(REXGRAPH_ALLOWED_FETCH_HOSTS)")


def check_db_uri(uri: str, *, allow_sql: bool = False) -> None:
    """Raise HTTPException(400) if ``uri`` violates the configured DB policy.

    SQL requires allow_sql=True at an explicit source operation. Deployment
    host/scheme policies still apply before driver access.
    """
    from agent.record_access import SQL_SCHEMES
    scheme = str(uri or "").split("://", 1)[0].split("+", 1)[0].lower()
    if scheme in SQL_SCHEMES and not allow_sql:
        raise HTTPException(400, "SQL access requires an explicit SQL source operation; Agent record storage uses RCDB")
    if not uri or "://" not in uri or not _policy_active():
        return

    parsed = urlparse(uri)
    base_scheme = (parsed.scheme or "").split("+", 1)[0].lower()
    host = parsed.hostname or ""

    allowed_schemes = _csv_env("REXGRAPH_ALLOWED_DB_SCHEMES")
    allowed_hosts = _csv_env("REXGRAPH_ALLOWED_DB_HOSTS")
    safe = (os.environ.get("REXGRAPH_DB_SAFE") == "1"
            or os.environ.get("REXGRAPH_DB_BLOCK_LOCAL") == "1")

    if allowed_schemes is not None and base_scheme not in allowed_schemes:
        raise HTTPException(
            400, f"database scheme '{base_scheme}' is not permitted "
                 f"(REXGRAPH_ALLOWED_DB_SCHEMES)")

    if safe and base_scheme in _FILE_SCHEMES:
        raise HTTPException(400, f"local source scheme '{base_scheme}' requires a file handle")

    if safe and _host_is_local_or_private(host):
        raise HTTPException(
            400, f"connection to loopback/private host '{host}' is blocked "
                 f"(REXGRAPH_DB_SAFE, anti-SSRF)")

    if allowed_hosts is not None and host.lower() not in allowed_hosts:
        raise HTTPException(
            400, f"database host '{host or '(none)'}' is not permitted "
                 f"(REXGRAPH_ALLOWED_DB_HOSTS)")
