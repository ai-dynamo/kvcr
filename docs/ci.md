# CI integration image

The **Docker Build** GitHub Actions workflow installs Dynamo nightly wheels and
KVCR on a prebuilt vLLM image using `Dockerfile.ci`. It runs automatically on
pushes to `pull-request/<number>` branches, which copy-pr-bot creates once a
maintainer approves a PR commit with `/ok to test <sha>`, using the defaults
below. To override them, use **Run workflow** in the Actions tab and supply:

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
`Dockerfile.ci`. It routes the build to the shared BuildKit service (the `vllm`
flavor), resolves the base's `linux/amd64` digest with a registry query,
installs Dynamo's wheels, and installs KVCR from that checkout. No Dynamo source compilation or vLLM source overlay is
required. Dynamo is installed without the `[vllm]` extra to keep the selected
backend, and constraints preserve the base's vLLM, torch, NumPy, and protobuf
versions. Only Dynamo's version resolution enables prereleases; installing the
resolved version does not enable prereleases for unrelated dependencies. The
image is built on BuildKit and pushed to ECR as
`210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/kvcr:<checked-out KVCR commit SHA>`;
it is never loaded into a local Docker daemon. The image labels record the KVCR
SHA and resolved base image. The installed
package versions and Dynamo's pip installation report (including downloaded
wheel URLs and hashes) are stored in the image under
`/opt/kvcr-build-metadata.json` and `/opt/kvcr-dynamo-install-report.json`. The
workflow does not extract them; a test job can read them from the running image.
The build preserves the base's installed NCCL version, rejects dependency
conflicts introduced by installation, and checks that Dynamo and the vLLM KVCR
adapter import against the newly installed package.
An older adapter can fail this check even when installing KVCR itself succeeds.
The inherited vLLM API-server entrypoint is cleared so tests can launch Dynamo.

Dynamo's [nightly wheel installation instructions](https://docs.nvidia.com/dynamo/latest/reference/nightly-releases)
describe the NVIDIA prerelease index. Floating nightly inputs can change between
builds; use the recorded base digest and installed Dynamo version for reproduction.

The job runs on `prod-kvcr-builder-v1`, a thin builder client with no Docker
daemon, so the workflow never pulls or runs images locally. Layers are built and
cached on the shared BuildKit pods. Private base images require registry
authentication before the **Resolve build inputs** step; no registry credentials
or infrastructure-specific authentication are configured yet.

The workflow logs in to ECR (`210086341041.dkr.ecr.us-west-2.amazonaws.com`)
with the runner's IRSA role and routes builds to BuildKit in the `buildkit`
namespace; neither needs a repository variable. The `ai-dynamo/kvcr`
repository is defined in velonix Terraform (`staging/ecr.tf`). Registry
authentication for private base images is not configured yet. A separate job on
`prod-kvcr-tester-amd-gpu-2-v1` can pull the SHA-tagged image and run the
two-rank serving test; that job is not part of this workflow yet.

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
