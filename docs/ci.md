# CI integration image

The **Docker Build** GitHub Actions workflow builds a thin KVCR layer on a
prebuilt Dynamo vLLM image using `Dockerfile.ci`. Use **Run workflow** in the
Actions tab and supply:

- `base_image`: a compatible Dynamo vLLM nightly or PR image, including its
  registry and tag or digest. This selects the Dynamo and vLLM revisions used
  by the test; they are inherited from the base rather than built from source.
- `kvcr_ref`: optionally, the KVCR ref to check out, such as
  `refs/pull/23/merge`. If omitted, the workflow builds its selected revision.

The workflow keeps the CI recipe from its selected revision and checks out the
requested KVCR source separately, so older PR refs do not need to contain
`Dockerfile.ci`. It sets up Docker Buildx, pulls the base for
`linux/amd64`, resolves its digest, and installs KVCR from that checkout. It
loads the resulting image into the local Docker daemon with the tag
`kvcr-vllm-e2e:<checked-out KVCR commit SHA>`. The image labels record the KVCR
SHA and resolved base image. A `kvcr-build-metadata` artifact records the same
inputs, including the originally requested base reference, for seven days.

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
export DYNAMO_IMAGE='<compatible-dynamo-vllm-image>'
export KVCR_COMMIT=$(git rev-parse HEAD)
docker buildx build --load --platform linux/amd64 \
  --file Dockerfile.ci \
  --build-arg DYNAMO_IMAGE="$DYNAMO_IMAGE" \
  --build-arg KVCR_COMMIT="$KVCR_COMMIT" \
  --tag "kvcr-vllm-e2e:$KVCR_COMMIT" \
  .
```
