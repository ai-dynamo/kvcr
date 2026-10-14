# CI integration image

The **Docker Build** GitHub Actions workflow installs Dynamo nightly wheels and
KVCR on a prebuilt vLLM image using `Dockerfile.ci`. Use **Run workflow** in the
Actions tab and supply:

- `base_image`: defaults to `vllm/vllm-openai:nightly`. Supply a compatible
  vLLM nightly or PR image tag or digest to select a specific backend build.
- `dynamo_version`: defaults to `nightly`, resolving the latest prerelease
  `ai-dynamo` wheel from NVIDIA's package index. Supply an exact version such
  as `1.6.0.dev20261014` to reproduce a specific nightly. Its dependency selects
  the matching `ai-dynamo-runtime` wheel.
- `kvcr_ref`: optionally, the KVCR ref to check out, such as
  `refs/pull/23/merge`. If omitted, the workflow builds its selected revision.

The workflow keeps the CI recipe from its selected revision and checks out the
requested KVCR source separately, so older PR refs do not need to contain
`Dockerfile.ci`. It sets up Docker Buildx, pulls the base for
`linux/amd64`, resolves its digest, installs Dynamo's wheels, and installs KVCR
from that checkout. No Dynamo source compilation or vLLM source overlay is
required. Dynamo is installed without the `[vllm]` extra to keep the selected
backend, and constraints preserve the base's vLLM, torch, NumPy, and protobuf
versions. Only Dynamo's version resolution enables prereleases; installing the
resolved version does not enable prereleases for unrelated dependencies. It
loads the resulting image into the local Docker daemon with the tag
`kvcr-vllm-e2e:<checked-out KVCR commit SHA>`. The image labels record the KVCR
SHA and resolved base image. A `kvcr-build-metadata` artifact records the same
inputs, the installed package versions, and Dynamo's pip installation report
(including downloaded wheel URLs and hashes) for seven days. These reports are
also stored in the image under `/opt/kvcr-*.json`.
The build preserves the base's installed NCCL version, rejects dependency
conflicts introduced by installation, and checks the Dynamo
router-hint imports and vLLM KVCR adapter against the newly installed package.
An older adapter can fail this check even when installing KVCR itself succeeds.
The inherited vLLM API-server entrypoint is cleared so tests can launch Dynamo.

Dynamo's [nightly wheel installation instructions](https://docs.nvidia.com/dynamo/latest/reference/nightly-releases)
describe the NVIDIA prerelease index. Floating nightly inputs can change between
builds; use the recorded base digest and installed Dynamo version for reproduction.

The build runs locally on `ubuntu-latest`. The runner needs enough disk for the
base image, build layers, and loaded result. Private base images require registry
authentication before the **Resolve build inputs** step; no registry credentials
or infrastructure-specific authentication are configured yet.

The **Build local integration image** step is the replacement point for routing
to the shared BuildKit service and publishing to ECR. Once registry and runner
access are configured, a separate job on `prod-kvcr-tester-amd-gpu-2-v1` can pull
the published SHA-tagged image and run the two-rank serving test. The current
workflow builds the image only; it does not publish it or run the GPU test, and
its local image disappears with the runner.

To build the same layer locally:

```bash
export VLLM_IMAGE=vllm/vllm-openai:nightly
export DYNAMO_VERSION=nightly
export KVCR_COMMIT=$(git rev-parse HEAD)
docker buildx build --load --platform linux/amd64 \
  --file Dockerfile.ci \
  --build-arg VLLM_IMAGE="$VLLM_IMAGE" \
  --build-arg DYNAMO_VERSION="$DYNAMO_VERSION" \
  --build-arg KVCR_COMMIT="$KVCR_COMMIT" \
  --tag "kvcr-vllm-e2e:$KVCR_COMMIT" \
  .
```
