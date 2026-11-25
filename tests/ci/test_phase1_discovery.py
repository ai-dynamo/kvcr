# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Readiness discovery retains evidence across a worker replacement, without GPUs."""

import json
import sys
import tempfile
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

from phase1.client import Client  # noqa: E402
from phase1.stack import NON_GUARD, Settings, Stack  # noqa: E402


@pytest.mark.parametrize("topology", ["agg", "disagg"])
def test_worker_rediscovery_preserves_original_probe_evidence(tmp_path, topology):
    """Run real discovery, file discovery and Client writes against a tiny HTTP peer."""
    observations = []
    prefill_id = 101
    decode_id = 303 if topology == "disagg" else prefill_id

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observations.append({"body": body, "headers": dict(self.headers)})
            payload = {
                "model": body["model"],
                "choices": [{"text": "ready", "finish_reason": "length"}],
                "nvext": {
                    "worker_id": {
                        "prefill_worker_id": prefill_id,
                        "prefill_dp_rank": 0,
                        "decode_worker_id": decode_id,
                        "decode_dp_rank": 0,
                    }
                },
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Probe-Worker", str(prefill_id))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    with ExitStack() as cleanup:
        server = HTTPServer(("127.0.0.1", 0), Endpoint)
        cleanup.callback(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cleanup.callback(thread.join, timeout=3)
        cleanup.callback(server.shutdown)
        subject = Stack(
            Settings(model="model", output=tmp_path, mode=NON_GUARD, topology=topology),
            tmp_path / "stack",
        )
        cleanup.callback(subject.cleanup)
        subject.discovery = tempfile.TemporaryDirectory(dir=tmp_path)
        records = Path(subject.discovery.name) / "v1" / "instances"
        records.mkdir(parents=True)
        evidence = tmp_path / "requests"
        evidence.mkdir()
        client = Client(
            f"http://127.0.0.1:{server.server_port}",
            "model",
            evidence,
            max_tokens=32,
            request_timeout=3,
        )
        if topology == "disagg":
            subject.worker_ids[2] = str(decode_id)

        def advertise(worker):
            for record in records.iterdir():
                record.unlink()
            (records / f"dynamo%2Fbackend%2Fgenerate%2F{worker:x}").write_text("{}")

        advertise(prefill_id)
        assert subject.discover_worker(client, 0) == "101"
        original_ledger = dict(client.log[0])
        original_request = (evidence / original_ledger["request_file"]).read_bytes()
        original_response = (evidence / original_ledger["response_file"]).read_bytes()
        original_wire = observations[0]

        subject.retired_ids.add(subject.worker_ids.pop(0))
        prefill_id = 202
        if topology == "agg":
            decode_id = prefill_id
        advertise(prefill_id)
        assert subject.discover_worker(client, 0) == "202"

        assert len(client.log) == 2
        assert len({entry["request_file"] for entry in client.log}) == 2
        assert len({entry["response_file"] for entry in client.log}) == 2
        assert client.log[0] == original_ledger
        assert (
            evidence / original_ledger["request_file"]
        ).read_bytes() == original_request
        assert (
            evidence / original_ledger["response_file"]
        ).read_bytes() == original_response
        for ledger, observed, expected_worker in zip(
            client.log, observations, (101, 202)
        ):
            request = json.loads((evidence / ledger["request_file"]).read_text())
            response = json.loads((evidence / ledger["response_file"]).read_text())
            assert (
                request["started_at"] == response["started_at"] == ledger["started_at"]
            )
            assert response["finished_at"] == ledger["finished_at"]
            assert request["body"] == observed["body"]
            assert request["body"]["prompt"].startswith("identity probe ")
            assert len(request["body"]["prompt"].split()[-1]) == 32
            assert request["body"]["max_tokens"] == 1
            assert request["body"]["nvext"]["extra_fields"] == ["worker_id"]
            assert (
                response["body"]["nvext"]["worker_id"]["prefill_worker_id"]
                == expected_worker
            )
            assert response["headers"]["X-Probe-Worker"] == str(expected_worker)
            if topology == "disagg":
                assert request["headers"]["x-dynamo-prefill-instance-id"] == str(
                    expected_worker
                )
                assert request["headers"]["x-dynamo-worker-instance-id"] == "303"
            else:
                assert request["headers"]["x-dynamo-worker-instance-id"] == str(
                    expected_worker
                )
        assert observations[0] == original_wire
        assert len(list(evidence.glob("*.request.json"))) == 2
        assert len(list(evidence.glob("*.response.json"))) == 2
