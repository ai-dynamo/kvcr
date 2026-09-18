# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline build-orchestration checks; Docker and upstream Git reads are mocked.

Run without importing KVCR or installing GPU dependencies:
    python3 -m unittest discover -s tests/ci -p 'test_*.py'
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/build_vllm_e2e.sh"
DYNAMO_SHA = "d" * 40
VLLM_SHA = "b" * 40
MANIFEST_DIGEST = "sha256:" + "a" * 64
CONFIG_DIGEST = "sha256:" + "c" * 64
BASE_IMAGE = "vllm/vllm-openai:fixture@sha256:" + "e" * 64

FAKE_GIT = r"""#!/usr/bin/env python3
import json, os, sys
arguments = sys.argv[1:]
if "ls-remote" in arguments:
    with open(os.environ["MOCK_GIT_CALLS"], "a") as stream:
        stream.write(json.dumps(arguments) + "\n")
    reference = arguments[-1]
    shas = {"refs/heads/main": "d" * 40, "refs/pull/53624/head": "b" * 40}
    if reference not in shas:
        raise SystemExit("Unexpected upstream reference")
    print(shas[reference] + "\t" + reference)
else:
    os.execv(os.environ["REAL_GIT"], [os.environ["REAL_GIT"], *arguments])
"""

FAKE_DOCKER = r"""#!/usr/bin/env python3
import json, os, pathlib, sys
arguments = sys.argv[1:]
with open(os.environ["MOCK_DOCKER_CALLS"], "a") as stream:
    stream.write(json.dumps(arguments) + "\n")
root = pathlib.Path(os.environ["E2E_BUILD_OUTPUT"])
if arguments[:2] == ["buildx", "build"]:
    metadata = pathlib.Path(arguments[arguments.index("--metadata-file") + 1])
    digests = {
        "containerimage.digest": "sha256:" + "a" * 64,
        "containerimage.config.digest": "sha256:" + "c" * 64,
    }
    if "MOCK_METADATA_JSON" in os.environ:
        digests = json.loads(os.environ["MOCK_METADATA_JSON"])
    metadata.write_text(json.dumps(digests))
    print("Mock image build; no Docker daemon or registry is contacted")
elif (arguments[:2] == ["image", "inspect"] or
      arguments[:3] == ["buildx", "imagetools", "inspect"]):
    manifest = json.loads((root / "source-revisions.json").read_text())
    labels = {
        "org.opencontainers.image.revision": manifest["kvcr_commit"],
        "io.ai-dynamo.kvcr.dynamo.sha": manifest["dynamo_commit"],
        "io.ai-dynamo.kvcr.vllm.sha": manifest["vllm_adapter_commit"],
        "io.ai-dynamo.kvcr.base-image": manifest["vllm_base_image"],
    }
    if os.environ.get("MOCK_WRONG_LABEL"):
        labels["io.ai-dynamo.kvcr.vllm.sha"] = "f" * 40
    if arguments[0] == "image":
        local_id = os.environ.get("MOCK_LOCAL_IMAGE_ID", "sha256:" + "c" * 64)
        print(json.dumps([{"Id": local_id, "Architecture": "amd64",
                           "Os": "linux", "Config": {"Labels": labels}}]))
    else:
        configuration = {
            "architecture": "amd64", "os": "linux", "config": {"Labels": labels}
        }
        if os.environ.get("MOCK_PLATFORM_MAP"):
            configuration = {"linux/amd64": configuration}
        print(json.dumps(configuration))
else:
    raise SystemExit("Unexpected Docker operation")
"""


class BuildVllmE2ETests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kvcr-build-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.real_git = shutil.which("git")
        self.assertIsNotNone(self.real_git)
        for args in (
            ["init", "--quiet"],
            ["config", "user.name", "KVCR Offline Test"],
            ["config", "user.email", "kvcr-test@example.invalid"],
        ):
            subprocess.run(
                [self.real_git, *args], cwd=self.repo, check=True, capture_output=True
            )
        (self.repo / "src").mkdir()
        (self.repo / "src/fixture.py").write_text("VALUE = 1\n")
        (self.repo / "Dockerfile.quick-start").write_text(
            f"ARG VLLM_IMAGE={BASE_IMAGE}\nFROM ${{VLLM_IMAGE}}\n"
        )
        for filename in ("pyproject.toml", "README.md", "LICENSE"):
            (self.repo / filename).write_text("offline fixture\n")
        subprocess.run(
            [self.real_git, "add", "."], cwd=self.repo, check=True, capture_output=True
        )
        subprocess.run(
            [self.real_git, "commit", "--quiet", "-m", "Offline fixture"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.sha = subprocess.check_output(
            [self.real_git, "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        binaries = self.root / "bin"
        binaries.mkdir()
        for name, contents in (("git", FAKE_GIT), ("docker", FAKE_DOCKER)):
            path = binaries / name
            path.write_text(contents)
            path.chmod(0o755)
        self.output = self.repo / "artifacts/build"
        self.github_output = self.root / "github-output"
        self.environment = dict(os.environ)
        # Never inherit CI identity or user-selected build settings into fixtures.
        for name in list(self.environment):
            if name.startswith(("GITHUB_", "KVCR_", "MOCK_")):
                self.environment.pop(name)
        self.environment.update(
            PATH=str(binaries) + os.pathsep + os.environ.get("PATH", ""),
            REAL_GIT=self.real_git,
            MOCK_GIT_CALLS=str(self.root / "git-calls.jsonl"),
            MOCK_DOCKER_CALLS=str(self.root / "docker-calls.jsonl"),
            E2E_BUILD_OUTPUT=str(self.output),
            GITHUB_SHA=self.sha,
            GITHUB_RUN_ID="42",
            GITHUB_RUN_ATTEMPT="3",
            GITHUB_REF="refs/pull/23/merge",
            GITHUB_EVENT_NAME="pull_request",
            GITHUB_REPOSITORY="ai-dynamo/kvcr",
            GITHUB_OUTPUT=str(self.github_output),
            KVCR_E2E_IMAGE="registry.example.invalid/kvcr/e2e",
        )

    def run_script(self, *arguments, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            cwd=self.repo,
            env=dict(self.environment, **overrides),
            capture_output=True,
            text=True,
            timeout=20,
        )

    def calls(self, kind):
        path = self.root / f"{kind}-calls.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def assert_failed_without_outputs(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.github_output.exists())
        self.assertFalse((self.output / "image-identity.json").exists())

    def test_resolve_only_records_each_upstream_once_without_docker(self):
        result = self.run_script("--resolve-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls("git")), 2)
        self.assertEqual(self.calls("docker"), [])
        manifest = json.loads((self.output / "source-revisions.json").read_text())
        self.assertEqual(manifest["dynamo_commit"], DYNAMO_SHA)
        self.assertEqual(manifest["vllm_adapter_commit"], VLLM_SHA)
        self.assertEqual(manifest["kvcr_commit"], self.sha)
        self.assertEqual(manifest["vllm_base_image"], BASE_IMAGE)
        self.assertEqual(manifest["github_sha"], self.sha)
        self.assertEqual(manifest["github_ref"], "refs/pull/23/merge")
        self.assertEqual(manifest["github_event_name"], "pull_request")
        self.assertEqual(manifest["github_run_id"], "42")
        self.assertEqual(manifest["github_run_attempt"], "3")
        self.assertEqual(manifest["github_repository"], "ai-dynamo/kvcr")
        self.assertFalse(self.github_output.exists())

    def test_github_sha_mismatch_fails_before_resolution_or_build(self):
        result = self.run_script(GITHUB_SHA="f" * 40)
        self.assert_failed_without_outputs(result)
        self.assertIn("does not match GITHUB_SHA", result.stderr)
        self.assertEqual(self.calls("git"), [])
        self.assertEqual(self.calls("docker"), [])

    def test_untracked_copied_source_fails_before_build(self):
        (self.repo / "src/new_module.py").write_text("UNTRACKED = True\n")
        result = self.run_script()
        self.assert_failed_without_outputs(result)
        self.assertIn("untracked copied source", result.stderr)
        self.assertEqual(self.calls("git"), [])
        self.assertEqual(self.calls("docker"), [])

    def test_local_load_preserves_generated_artifacts_and_checks_labels(self):
        self.output.mkdir(parents=True)
        (self.output / "previous-note.txt").write_text("generated artifact\n")
        result = self.run_script(
            KVCR_PUSH="0", KVCR_BUILDER="fixture-builder", KVCR_GITHUB_IP="140.82.116.3"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        build = self.calls("docker")[0]
        self.assertIn("--load", build)
        self.assertEqual(build[build.index("--ulimit") + 1], "nofile=65536:65536")
        self.assertNotIn("--push", build)
        self.assertNotIn("--secret", build)
        self.assertIn("KVCR_RUNTIME_USER=root", build)
        self.assertIn("DYNAMO_REF=" + DYNAMO_SHA, build)
        self.assertIn("KVCR_VLLM_REF=" + VLLM_SHA, build)
        self.assertIn("fixture-builder", build)
        self.assertIn("github.com:140.82.116.3", build)
        identity = json.loads((self.output / "image-identity.json").read_text())
        self.assertFalse(identity["published"])
        self.assertEqual(identity["local_image_id"], CONFIG_DIGEST)
        self.assertTrue(identity["image_uri"].endswith(self.sha + "-run-42-3"))
        self.assertEqual(len(self.calls("git")), 2)

    def test_containerd_load_accepts_manifest_id_without_config_digest(self):
        result = self.run_script(
            KVCR_PUSH="0",
            MOCK_METADATA_JSON=json.dumps({"containerimage.digest": MANIFEST_DIGEST}),
            MOCK_LOCAL_IMAGE_ID=MANIFEST_DIGEST,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads((self.output / "image-identity.json").read_text())
        self.assertEqual(identity["local_image_id"], MANIFEST_DIGEST)
        self.assertEqual(identity["digest"], MANIFEST_DIGEST)

    def test_classic_load_accepts_config_only_metadata(self):
        result = self.run_script(
            KVCR_PUSH="0",
            MOCK_METADATA_JSON=json.dumps(
                {"containerimage.config.digest": CONFIG_DIGEST}
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads((self.output / "image-identity.json").read_text())
        self.assertEqual(identity["local_image_id"], CONFIG_DIGEST)
        self.assertEqual(identity["digest"], CONFIG_DIGEST)

    def test_local_load_rejects_unrecorded_id_or_missing_digests(self):
        for metadata in (
            {"containerimage.digest": MANIFEST_DIGEST},
            {
                "containerimage.digest": MANIFEST_DIGEST,
                "containerimage.config.digest": CONFIG_DIGEST,
            },
            {},
        ):
            with self.subTest(metadata=metadata):
                result = self.run_script(
                    KVCR_PUSH="0",
                    MOCK_METADATA_JSON=json.dumps(metadata),
                    MOCK_LOCAL_IMAGE_ID="sha256:" + "f" * 64,
                )
                self.assert_failed_without_outputs(result)
                self.assertIn("does not match any recorded build digest", result.stderr)

    def test_invalid_metadata_or_local_id_rejects_outputs(self):
        cases = (
            {"MOCK_LOCAL_IMAGE_ID": "invalid"},
            {
                "MOCK_METADATA_JSON": json.dumps(
                    {
                        "containerimage.digest": MANIFEST_DIGEST,
                        "containerimage.config.digest": "invalid",
                    }
                )
            },
            {"MOCK_METADATA_JSON": json.dumps({"containerimage.digest": 123})},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                result = self.run_script(KVCR_PUSH="0", **overrides)
                self.assert_failed_without_outputs(result)
                self.assertIn("invalid", result.stderr)

    def test_mock_publish_emits_only_verified_digest_reference(self):
        result = self.run_script(KVCR_PUSH="1", MOCK_PLATFORM_MAP="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--push", self.calls("docker")[0])
        output = self.github_output.read_text()
        self.assertIn(
            "image_uri=registry.example.invalid/kvcr/e2e@" + MANIFEST_DIGEST, output
        )
        self.assertIn("kvcr_sha=" + self.sha, output)
        self.assertIn("dynamo_sha=" + DYNAMO_SHA, output)
        self.assertIn("vllm_sha=" + VLLM_SHA, output)
        identity = json.loads((self.output / "image-identity.json").read_text())
        self.assertNotIn("local_image_id", identity)

    def test_wrong_provenance_label_rejects_outputs(self):
        result = self.run_script(KVCR_PUSH="0", MOCK_WRONG_LABEL="1")
        self.assert_failed_without_outputs(result)
        self.assertIn("provenance label does not match", result.stderr)

    def test_invalid_dns_override_never_invokes_docker(self):
        result = self.run_script(KVCR_GITHUB_IP="not-an-ip")
        self.assert_failed_without_outputs(result)
        self.assertEqual(self.calls("docker"), [])
        self.assertIn("numeric IPv4", result.stderr)


if __name__ == "__main__":
    unittest.main()
