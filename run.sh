#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="webcast-container"
IMAGE_NAME="webcast"
PORT_MAP="8501:8501"
ENV_FILE=".env"
HOST_DIR="$(pwd)"
CONTAINER_DIR="/app"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}:latest" >/dev/null 2>&1; then
  echo "Image '${IMAGE_NAME}:latest' not found. Building..."
  docker build -t "${IMAGE_NAME}" .
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Warning: ${ENV_FILE} not found; continuing without --env-file."
  ENV_ARGS=()
else
  ENV_ARGS=(--env-file "$ENV_FILE")
fi

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "Stopping running container: ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

docker run --rm \
  --name "${CONTAINER_NAME}" \
  "${ENV_ARGS[@]}" \
  -p "${PORT_MAP}" \
  -v "${HOST_DIR}:${CONTAINER_DIR}" \
  "${IMAGE_NAME}"
