#!/bin/bash
set -euo pipefail

# Build and push the CI Docker image.
#
# Uses a pre-built deps base image (tagged by content hash) so that most
# builds only need to COPY source code on top — no pip install, no container
# driver overhead. The deps base is rebuilt automatically when dependencies change.
#
# Expects the following environment variable (set by Buildkite):
#   BUILDKITE_COMMIT          - commit SHA used as the image tag
ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"

# Ensure that the env vars are actually set, otherwise exit early
if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "ERROR: BUILDKITE_COMMIT is not set"
    exit 1
fi
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"

# Compute deps image tag from dependency file contents
DEPS_KEY=$(cat pyproject.toml setup.py requirements/*.txt | sha256sum | cut -c1-16)
DEPS_TAG="ci-deps-${DEPS_KEY}"
DEPS_IMAGE="${REGISTRY}:${DEPS_TAG}"
echo "Deps image: ${DEPS_IMAGE}"

# Authenticate to ECR Public
aws ecr-public get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_NAMESPACE"

# Build deps base image if it doesn't exist
if docker manifest inspect "${DEPS_IMAGE}" >/dev/null 2>&1; then
    echo "Deps image exists: ${DEPS_TAG}"
else
    echo "--- :docker: Building deps base image (cache miss: ${DEPS_TAG})"
    docker build --progress=plain \
        --target deps \
        --file docker/Dockerfile.ci \
        -t "${DEPS_IMAGE}" .
    docker push "${DEPS_IMAGE}"
fi

# Build the CI image on top of the deps base
echo "--- :docker: Building CI image"
echo "Image tag: ${REGISTRY}:${BUILDKITE_COMMIT}"

docker build --progress=plain \
    --build-arg "CI_DEPS_IMAGE=${DEPS_IMAGE}" \
    --file docker/Dockerfile.ci \
    -t "${REGISTRY}:${BUILDKITE_COMMIT}" .
docker push "${REGISTRY}:${BUILDKITE_COMMIT}"
