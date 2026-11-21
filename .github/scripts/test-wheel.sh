#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: test-wheel.sh <wheel> <version> <nightly|release>. Runs third-party code, so never call it in a job holding release credentials.
set -euo pipefail

WHEEL="$1"
VERSION="$2"
MODE="$3"
export VERSION
TWINE_VERSION="7.0.0"
GH_VERSION="2.103.3"
GH_SHA256="4c806978ffce1cec70853302a56d70ac6da4f19bae7adca4e1a8ab00633dd91d"

uvx "twine==${TWINE_VERSION}" check --strict "${WHEEL}"

if ! command -v gh > /dev/null; then
  tarball="${RUNNER_TEMP}/gh.tar.gz"
  curl -q -fsSL --retry 3 --max-time 120 -o "${tarball}" \
    "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz"
  echo "${GH_SHA256}  ${tarball}" | sha256sum --check
  tar -xzf "${tarball}" -C "${RUNNER_TEMP}"
  export PATH="${RUNNER_TEMP}/gh_${GH_VERSION}_linux_amd64/bin:${PATH}"
fi

# kvcr loads the system libatomic at runtime, and the kvcr builder image lacks it.
if ! /sbin/ldconfig -p | grep -q 'libatomic\.so\.1'; then
  sudo=""; [ "$(id -u)" = "0" ] || sudo="sudo"
  ${sudo} apt-get update -q
  ${sudo} apt-get install -y -q --no-install-recommends libatomic1
fi

for py in 3.10 3.12; do
  venv="${RUNNER_TEMP}/venv-${py}"
  uv venv --python "${py}" "${venv}"
  uv pip install --python "${venv}/bin/python" "${WHEEL}[profiling]"
  uv pip check --python "${venv}/bin/python"
  # Inventory before pytest so test tooling stays out of the license evidence.
  python3 .github/scripts/wheel_compliance.py licenses --python "${venv}/bin/python" \
    --inventory "${RUNNER_TEMP}/inventory/py${py}.csv"
  "${venv}/bin/python" -c 'import importlib.metadata as m, os, sys, kvcr; assert m.version("kvcr") == os.environ["VERSION"]; assert kvcr.__file__.startswith(sys.prefix), kvcr.__file__'
  "${venv}/bin/kvcr-service" --help > /dev/null
  uv pip install --python "${venv}/bin/python" "pytest>=8,<9"
  "${venv}/bin/python" -m pytest tests/unit -q -p no:cacheprovider
done

# Nightlies skip the run status, which an advisory GPU failure turns red; evidence only exists after a passed license check.
if [ "${MODE}" = "release" ]; then
  query="actions/workflows/release.yml/runs?event=workflow_dispatch&status=success&per_page=20"
else
  query="actions/workflows/nightly.yml/runs?branch=main&event=schedule&per_page=20"
fi
prior=()
baseline="none found; every dependency is listed as added"
runs="$(gh api "repos/${GITHUB_REPOSITORY}/${query}" --jq '.workflow_runs[].id')" || runs=""
for run in ${runs}; do
  [ "${run}" = "${GITHUB_RUN_ID}" ] && continue
  dir="${RUNNER_TEMP}/prior-${run}"
  if gh run download "${run}" -R "${GITHUB_REPOSITORY}" -n license-artifacts -D "${dir}" 2> /dev/null \
    && [ -f "${dir}/deps.csv" ]; then
    prior=(--prior "${dir}/deps.csv")
    baseline="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${run}"
    break
  fi
done
echo "Diff baseline: ${baseline}"
out="${RUNNER_TEMP}/license-artifacts"
python3 .github/scripts/wheel_compliance.py evidence "${RUNNER_TEMP}"/inventory/py*.csv \
  --out "${out}" ${prior[@]+"${prior[@]}"}
mkdir -p "${out}/baseline"
printf '# Diff baseline\n\nThis run: %s\nCompared with: %s\n' \
  "${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}" "${baseline}" > "${out}/baseline/BASELINE.md"
