#!/bin/bash
set -euo pipefail

# Build and push the CI Docker image with registry-based layer caching.
#
# Usage: build-ci-image.sh
#
# Expects the following environment variables (set by Buildkite):
#   BUILDKITE_COMMIT          - commit SHA used as the image tag
ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"
DOCKERFILE="docker/Dockerfile.ci"

# Files that determine the install layer; we use the hash of the contents
# Of these files as the tag so that we can share the deps across most PRs.
DEP_FILES="pyproject.toml setup.py requirements/"

# Ensure that the env vars are actually set, otherwise exit early
if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "ERROR: BUILDKITE_COMMIT is not set"
    exit 1
fi
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"

# Compute cache tag from dependency file contents
CACHE_KEY=$(cat pyproject.toml setup.py requirements/*.txt | sha256sum | cut -c1-16)
CACHE_TAG="deps-cache-${CACHE_KEY}"
echo "Cache key: ${CACHE_TAG}"

# Authenticate to ECR Public
aws ecr-public get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_NAMESPACE"

# Set up buildx with docker-container driver; we need to do this
# since cache export is not supported for the default docker driver.
docker buildx create --name vllm-omni-builder --driver docker-container --use

echo "--- :docker: Building CI image"
echo "Image tag: ${REGISTRY}:${BUILDKITE_COMMIT}"

# Build + push the image to the registry, using content-hashed cache.
# Cache is shared across all branches/PRs with the same dependencies.
docker buildx build --push --progress=plain \
    --cache-from "type=registry,ref=${REGISTRY}:${CACHE_TAG}" \
    --cache-to "type=registry,ref=${REGISTRY}:${CACHE_TAG},mode=max,compression=zstd" \
    --file "$DOCKERFILE" \
    -t "${REGISTRY}:${BUILDKITE_COMMIT}" .
