#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release compliance checks for the kvcr wheel and sdist."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

# Dependencies cleared by open-source review; a new one needs its own review.
APPROVED_DEPENDENCIES = {"msgspec", "pyzmq", "nixl"}
APPROVED_EXTRA_DEPENDENCIES: set[str] = set()
COPYRIGHT = re.compile(
    r"# SPDX-FileCopyrightText: Copyright \(c\) \d{4}(-\d{4})? "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)
SPDX_ID = "# SPDX-License-Identifier: Apache-2.0"
ALLOWED_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC0-1.0",
    "ISC",
    "LLVM-exception",
    "MIT",
    "MIT-CMU",
    "MPL-2.0",
    "PSF-2.0",
    "Python-2.0",
    "Unlicense",
    "Zlib",
    "Apache Software License",
    "BSD License",
    "ISC License (ISCL)",
    "MIT License",
    "Mozilla Public License 2.0 (MPL 2.0)",
    "Python Software Foundation License",
    "The Unlicense (Unlicense)",
    "Apache 2.0",
    "BSD",
}
NVIDIA_PACKAGE = re.compile(r"^(nvidia|cuda)-")
NVIDIA_LICENSES = {
    "LicenseRef-NVIDIA-Proprietary",
    "NVIDIA Proprietary Software",
    "Other/Proprietary License",
    "UNKNOWN",
}
FIELDS = ["dependency_type", "name", "version", "spdx_license"]
DIFF_FIELDS = ["change", *FIELDS, "prior_version", "prior_spdx_license"]
DUMP = """
import importlib.metadata as m, json
print(json.dumps([{
    "name": d.metadata["Name"], "version": d.version,
    "expression": d.metadata.get("License-Expression") or "",
    "license": d.metadata.get("License") or "",
    "classifiers": d.metadata.get_all("Classifier") or [],
} for d in m.distributions()]))
"""


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def license_text(dist: dict) -> str:
    if dist["expression"].strip():
        return dist["expression"].strip()
    classifiers = [
        c.split(" :: ")[-1] for c in dist["classifiers"] if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)
    first = dist["license"].strip().splitlines()[:1]
    return first[0].strip() if first and len(first[0]) <= 80 else "UNKNOWN"


def license_terms(text: str) -> set[str]:
    parts = re.split(r"\s*;\s*|\s+(?:AND|OR|WITH)\s+|[()]", text)
    return {part.strip() for part in parts if part.strip()}


def check_wheel(
    wheel: Path, version: str, license_file: Path, strict: bool
) -> list[str]:
    errors = []
    dist_info = f"kvcr-{version}.dist-info/"
    required = {
        f"{dist_info}{name}"
        for name in ("METADATA", "WHEEL", "RECORD", "licenses/LICENSE")
    }
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            if name in required:
                continue
            if not name.startswith("kvcr/"):
                errors.append(f"{name}: unexpected file outside the kvcr package")
            elif not name.endswith(".py"):
                errors.append(f"{name}: only Python sources may ship in the wheel")
            else:
                head = archive.read(name).decode("utf-8").splitlines()[:3]
                if (
                    not any(COPYRIGHT.fullmatch(line) for line in head)
                    or SPDX_ID not in head
                ):
                    errors.append(
                        f"{name}: missing the NVIDIA SPDX copyright or license header"
                    )
        errors.extend(
            f"{name}: missing from the wheel" for name in sorted(required - set(names))
        )
        if errors:
            return errors
        metadata = Parser().parsestr(
            archive.read(f"{dist_info}METADATA").decode("utf-8")
        )
        if archive.read(f"{dist_info}licenses/LICENSE") != license_file.read_bytes():
            errors.append("the wheel's LICENSE differs from the repository LICENSE")
    found = f"{metadata['Name']} {metadata['Version']}"
    if found != f"kvcr {version}":
        errors.append(f"METADATA names {found}, expected kvcr {version}")
    expression = metadata["License-Expression"]
    if expression != "Apache-2.0":
        errors.append(f"License-Expression is {expression!r}, expected 'Apache-2.0'")
    runtime, extras = set(), set()
    for req in metadata.get_all("Requires-Dist") or []:
        name = normalize(re.match(r"[A-Za-z0-9._-]+", req).group(0))
        marker = req.partition(";")[2]
        (extras if re.search(r"\bextra\s*==", marker) else runtime).add(name)
    unapproved = sorted(runtime - APPROVED_DEPENDENCIES)
    if unapproved:
        errors.append(
            f"runtime dependencies without open-source review: {', '.join(unapproved)}"
        )
    unreviewed = sorted(extras - APPROVED_DEPENDENCIES - APPROVED_EXTRA_DEPENDENCIES)
    if unreviewed:
        message = (
            f"optional dependencies without open-source review: {', '.join(unreviewed)}"
        )
        if strict:
            errors.append(message)
        else:
            print(f"::warning::{message}")
    return errors


def check_sdist(sdist: Path, version: str, license_file: Path) -> list[str]:
    root = f"kvcr-{version}"
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        tops = sorted({m.name.split("/")[0] for m in members})
        if tops != [root]:
            return [f"the sdist's top-level entries are {tops}, expected [{root!r}]"]
        errors = [
            f"{m.name}: only regular files and directories may ship in the sdist"
            for m in members
            if not (m.isfile() or m.isdir()) or ".." in m.name.split("/")
        ]
        if errors:
            return errors
        names = {m.name for m in members}
        missing = [f"{root}/{n}" for n in ("PKG-INFO", "LICENSE")]
        missing = [name for name in missing if name not in names]
        if missing:
            return [f"{name}: missing from the sdist" for name in missing]
        metadata = Parser().parsestr(
            archive.extractfile(f"{root}/PKG-INFO").read().decode("utf-8")
        )
        license_text = archive.extractfile(f"{root}/LICENSE").read()
    found = f"{metadata['Name']} {metadata['Version']}"
    if found != f"kvcr {version}":
        errors.append(f"PKG-INFO names {found}, expected kvcr {version}")
    expression = metadata["License-Expression"]
    if expression != "Apache-2.0":
        errors.append(f"License-Expression is {expression!r}, expected 'Apache-2.0'")
    if license_text != license_file.read_bytes():
        errors.append("the sdist's LICENSE differs from the repository LICENSE")
    return errors


def check_licenses(python: str, inventory: Path) -> list[str]:
    dists = json.loads(
        subprocess.run(
            [python, "-c", DUMP], check=True, capture_output=True, text=True
        ).stdout
    )
    rows, errors = {}, []
    for dist in dists:
        name = normalize(dist["name"])
        if name == "kvcr":
            continue
        text = license_text(dist)
        allowed = ALLOWED_LICENSES | (
            NVIDIA_LICENSES if NVIDIA_PACKAGE.match(name) else set()
        )
        if license_terms(text) - allowed:
            errors.append(
                f"{name} {dist['version']}: license {text!r} is not on the allowlist"
            )
        rows[name] = {
            "dependency_type": "python",
            "name": name,
            "version": dist["version"],
            "spdx_license": text,
        }
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))
    return errors


def read_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (r["dependency_type"], r["name"], r["version"]): r
            for r in csv.DictReader(stream)
        }


def write_evidence(
    inventories: list[Path], outdir: Path, prior: Path | None
) -> list[str]:
    current = {}
    for inventory in inventories:
        for key, row in read_rows(inventory).items():
            if current.setdefault(key, row) != row:
                return [f"{key[1]} {key[2]}: conflicting license metadata"]
    if not current:
        return ["no dependency inventories to merge"]
    previous = read_rows(prior) if prior and prior.is_file() else {}
    changes = []
    for key in sorted(set(current) | set(previous)):
        now, before = current.get(key), previous.get(key)
        if now and not before:
            changes.append(
                {
                    "change": "added",
                    **now,
                    "prior_version": "",
                    "prior_spdx_license": "",
                }
            )
        elif before and not now:
            changes.append(
                {
                    "change": "removed",
                    "dependency_type": before["dependency_type"],
                    "name": before["name"],
                    "version": "",
                    "spdx_license": "",
                    "prior_version": before["version"],
                    "prior_spdx_license": before["spdx_license"],
                }
            )
        elif now["spdx_license"] != before["spdx_license"]:
            changes.append(
                {
                    "change": "changed",
                    **now,
                    "prior_version": before["version"],
                    "prior_spdx_license": before["spdx_license"],
                }
            )
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in (
        ("deps.csv", FIELDS, [current[k] for k in sorted(current)]),
        ("deps-diff.csv", DIFF_FIELDS, changes),
    ):
        with (outdir / name).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"deps.csv: {len(current)} rows; deps-diff.csv: {len(changes)} rows"
        + ("" if previous else " (no earlier nightly to compare)")
    )
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wheel = commands.add_parser(
        "wheel", help="Check the wheel's contents, headers, license, and dependencies"
    )
    wheel.add_argument("wheel", type=Path)
    wheel.add_argument("--version", required=True)
    wheel.add_argument("--license-file", type=Path, default=Path("LICENSE"))
    wheel.add_argument(
        "--strict",
        action="store_true",
        help="Fail, not warn, on optional dependencies without open-source review",
    )
    sdist = commands.add_parser(
        "sdist", help="Check the sdist's layout, metadata, and license"
    )
    sdist.add_argument("sdist", type=Path)
    sdist.add_argument("--version", required=True)
    sdist.add_argument("--license-file", type=Path, default=Path("LICENSE"))
    licenses = commands.add_parser(
        "licenses", help="Check and inventory the licenses installed in an environment"
    )
    licenses.add_argument("--python", required=True)
    licenses.add_argument("--inventory", type=Path, required=True)
    evidence = commands.add_parser(
        "evidence", help="Write deps.csv and deps-diff.csv as license evidence"
    )
    evidence.add_argument("inventories", type=Path, nargs="+")
    evidence.add_argument("--out", type=Path, required=True)
    evidence.add_argument("--prior", type=Path)
    args = parser.parse_args()
    if args.command == "wheel":
        errors = check_wheel(args.wheel, args.version, args.license_file, args.strict)
    elif args.command == "sdist":
        errors = check_sdist(args.sdist, args.version, args.license_file)
    elif args.command == "licenses":
        errors = check_licenses(args.python, args.inventory)
    else:
        errors = write_evidence(args.inventories, args.out, args.prior)
    for error in errors:
        print(f"::error::{error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
