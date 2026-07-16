#!/bin/bash
set -euo pipefail

# Build and push the CI Docker image using buildx with registry-based
# layer caching.
#
# Uses a pre-built deps base image (tagged by content hash) so that most
# builds only need to COPY source code on top. When the deps hash hasn't
# changed, the deps stage is skipped entirely.
#
# Layer caching via --cache-from/--cache-to stores every intermediate
# layer in ECR (mode=max) so that even partial changes (e.g. adding one
# apt package) get fine-grained cache hits instead of rebuilding the
# whole stage. Multiple cache sources are tried in order:
#   1. Current commit (exact match from a previous run)
#   2. Parent commit (HEAD~1, likely close)
#   3. Deps content hash (matches any build with same deps)
#   4. "latest" (fallback to last successful main build)
#
# Inspired by the upstream vLLM CI caching system:
#   https://github.com/vllm-project/ci-infra/blob/main/docker/ci.hcl
#   https://github.com/vllm-project/ci-infra/blob/main/buildkite/scripts/ci-bake.sh
#
# Environment variables (set by Buildkite):
#   BUILDKITE_COMMIT  - commit SHA used as the image tag (required)
#
# Optional overrides:
#   CACHE_REGISTRY    - registry for layer cache (default: private ECR)
#   CACHE_REPO        - cache repo name (default: vllm-omni-ci-cache)

ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"
PRIVATE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com"
CACHE_REGISTRY="${CACHE_REGISTRY:-${PRIVATE_ECR}}"
CACHE_REPO="${CACHE_REPO:-vllm-omni-ci-cache}"
CACHE_IMAGE="${CACHE_REGISTRY}/${CACHE_REPO}"
BUILDER_NAME="vllm-omni-builder"

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "ERROR: BUILDKITE_COMMIT is not set"
    exit 1
fi
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"

# --- Check if final image already exists (skip entire build)
if docker manifest inspect "${REGISTRY}:${BUILDKITE_COMMIT}" >/dev/null 2>&1; then
    echo "Image already exists: ${REGISTRY}:${BUILDKITE_COMMIT}"
    echo "Skipping build"
    exit 0
fi

# --- Compute deps image tag from dependency file contents
DEPS_KEY=$(cat pyproject.toml setup.py requirements/*.txt | sha256sum | cut -c1-16)
DEPS_TAG="ci-deps-${DEPS_KEY}"
DEPS_IMAGE="${REGISTRY}:${DEPS_TAG}"
echo "Deps image: ${DEPS_IMAGE}"

# --- Compute parent commit for cache fallback
PARENT_COMMIT="${PARENT_COMMIT:-$(git rev-parse HEAD~1 2>/dev/null || echo "")}"
if [ -n "${PARENT_COMMIT}" ]; then
    echo "Parent commit (cache fallback): ${PARENT_COMMIT}"
fi

# --- Authenticate to ECR
aws ecr-public get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_NAMESPACE"

# Also auth to private ECR for cache storage
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$PRIVATE_ECR" 2>/dev/null || \
    echo "Warning: could not authenticate to private ECR for cache; layer caching disabled"

# --- Set up buildx builder
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
    echo "--- :buildkite: Setting up buildx builder"
    docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
else
    docker buildx use "${BUILDER_NAME}"
fi
docker buildx inspect --bootstrap

# --- Build cache-from sources (tried in order, first hit wins per layer)
CACHE_FROM_ARGS=()
CACHE_FROM_ARGS+=(--cache-from "type=registry,ref=${CACHE_IMAGE}:${BUILDKITE_COMMIT}")
[ -n "${PARENT_COMMIT}" ] && \
    CACHE_FROM_ARGS+=(--cache-from "type=registry,ref=${CACHE_IMAGE}:${PARENT_COMMIT}")
CACHE_FROM_ARGS+=(--cache-from "type=registry,ref=${CACHE_IMAGE}:${DEPS_TAG}")
CACHE_FROM_ARGS+=(--cache-from "type=registry,ref=${CACHE_IMAGE}:latest")

# --- Build deps base image if it doesn't exist in the public registry
if docker manifest inspect "${DEPS_IMAGE}" >/dev/null 2>&1; then
    echo "--- :white_check_mark: Deps image exists: ${DEPS_TAG}"
else
    echo "--- :docker: Building deps base image (cache miss: ${DEPS_TAG})"
    docker buildx build --progress=plain \
        --target deps \
        --file docker/Dockerfile.ci \
        "${CACHE_FROM_ARGS[@]}" \
        --cache-to "type=registry,ref=${CACHE_IMAGE}:${DEPS_TAG},mode=max,compression=zstd" \
        --output "type=registry" \
        -t "${DEPS_IMAGE}" .
fi

# --- Build the CI image on top of the deps base
echo "--- :docker: Building CI image"
echo "Image tag: ${REGISTRY}:${BUILDKITE_COMMIT}"

docker buildx build --progress=plain \
    --build-arg "CI_DEPS_IMAGE=${DEPS_IMAGE}" \
    --file docker/Dockerfile.ci \
    "${CACHE_FROM_ARGS[@]}" \
    --cache-to "type=registry,ref=${CACHE_IMAGE}:${BUILDKITE_COMMIT},mode=max,compression=zstd" \
    --output "type=registry" \
    -t "${REGISTRY}:${BUILDKITE_COMMIT}" .

# Update "latest" cache tag on main branch builds
if [ "${BUILDKITE_BRANCH:-}" = "main" ]; then
    echo "--- :label: Updating latest cache tag"
    docker buildx imagetools create \
        "${CACHE_IMAGE}:${BUILDKITE_COMMIT}" \
        --tag "${CACHE_IMAGE}:latest" 2>/dev/null || true
fi

echo "--- :white_check_mark: Build complete"
