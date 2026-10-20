"""Outbound network guard: only public http(s) destinations.

Used by the page fetcher (07, 12) and webhook delivery (03d). Every redirect hop
is re-checked, so a public URL cannot bounce the server to an internal address
(cloud metadata, localhost, private ranges).
"""

import asyncio
import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlsplit

Resolver = Callable[[str], list[str]]


class BlockedDestination(ValueError):
    pass


def _system_resolve(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)})


def check_public_url(url: str, *, resolve: Resolver = _system_resolve, require_https: bool = False) -> None:
    parts = urlsplit(url)
    allowed = ("https",) if require_https else ("http", "https")
    if parts.scheme not in allowed:
        raise BlockedDestination(f"scheme {parts.scheme!r} not allowed")
    host = parts.hostname
    if not host:
        raise BlockedDestination("URL has no host")
    try:
        addresses = [host] if _is_ip(host) else resolve(host)
    except OSError as exc:
        raise BlockedDestination(f"cannot resolve {host}") from exc
    if not addresses:
        raise BlockedDestination(f"cannot resolve {host}")
    for addr in addresses:
        if not ipaddress.ip_address(addr).is_global:
            raise BlockedDestination(f"{host} resolves to non-public address {addr}")


async def acheck_public_url(url: str, *, resolve: Resolver = _system_resolve, require_https: bool = False) -> None:
    await asyncio.to_thread(check_public_url, url, resolve=resolve, require_https=require_https)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
