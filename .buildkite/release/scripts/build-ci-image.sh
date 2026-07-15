#!/bin/bash
set -euo pipefail

# Build and push the CI Docker image with registry-based layer caching.
#
# Usage: build-ci-image.sh
#
# Expects the following environment variables (set by Buildkite):
#   BUILDKITE_COMMIT          - commit SHA used as the image tag
#   BUILDKITE_PULL_REQUEST    - PR number, or "false" if not a PR
ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"
DOCKERFILE="docker/Dockerfile.ci"

# Ensure that the env vars are actually set, otherwise exit early
MISSING=()
for var in BUILDKITE_COMMIT BUILDKITE_PULL_REQUEST; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Required environment variable(s) not set: ${MISSING[*]}"
    exit 1
fi
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"
echo "BUILDKITE_PULL_REQUEST: ${BUILDKITE_PULL_REQUEST}"


# Authenticate to ECR Public
aws ecr-public get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_NAMESPACE"

# Configure cache refs based on PR vs main branch
CACHE_FROM="--cache-from type=registry,ref=${REGISTRY}:cache-main"

if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
    BRANCH_CACHE="cache-pr-${BUILDKITE_PULL_REQUEST}"
    # It's a PR, so we also have the cache tag based on the PR to pull from
    CACHE_FROM_BRANCH="--cache-from type=registry,ref=${REGISTRY}:${BRANCH_CACHE}"
    CACHE_FROM="${CACHE_FROM} ${CACHE_FROM_BRANCH}"
else
    BRANCH_CACHE="cache-main"
fi
# Write the cache back to the main cache or the PR cache
CACHE_TO="--cache-to type=registry,ref=${REGISTRY}:${BRANCH_CACHE},mode=max"


echo "--- :docker: Building CI image"
echo "Image tag: ${REGISTRY}:${BUILDKITE_COMMIT}"
echo "Cache from: ${CACHE_FROM}"
echo "Cache to: ${CACHE_TO}"

# Build + push the build image to the registry (i.e., don't unpack on the CI machine)
docker buildx build --push --progress=plain \
    $CACHE_FROM $CACHE_TO \
    --file "$DOCKERFILE" \
    -t "${REGISTRY}:${BUILDKITE_COMMIT}" .
