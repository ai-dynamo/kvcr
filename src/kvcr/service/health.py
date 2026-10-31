# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional, read-only HTTP probes for the standalone service."""

import contextlib
import http.client
import threading
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextlib.contextmanager
def _serve_health(
    address: tuple[str, int] | None,
    status: Callable[[bool], tuple[int, str]],
    dependency_url: str | None = None,
) -> Iterator[ThreadingHTTPServer | None]:
    if address is None:
        yield None
        return

    # Loopback dependencies must not be redirected through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    class Handler(BaseHTTPRequestHandler):
        timeout = 2

        def do_GET(self) -> None:
            if self.path not in ("/live", "/ready"):
                code, reason = 404, "not found"
            else:
                ready = self.path == "/ready"
                code, reason = status(ready)
                if ready and code == 200 and dependency_url is not None:
                    try:
                        with opener.open(dependency_url, timeout=1) as response:
                            if response.status != 200:
                                code, reason = 503, "dependency unavailable"
                    except (OSError, http.client.HTTPException, ValueError):
                        code, reason = 503, "dependency unavailable"
            body = (reason + "\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            # Kubernetes probes are frequent; lifecycle failures are logged by owners.
            pass

    with ThreadingHTTPServer(address, Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, name="kvcr-health", daemon=True
        )
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()
