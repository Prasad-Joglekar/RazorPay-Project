#!/usr/bin/env bash
# Deploy the live console to Google Cloud Run.
#
# Every flag below is a decision rather than a default, and the two that matter
# most are --max-instances=1 and --timeout=3600. See deploy/cloudrun/DEPLOY.md.
set -euo pipefail

SERVICE="${SERVICE:-fraud-console}"
REGION="${REGION:-asia-south1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed: https://cloud.google.com/sdk/docs/install" >&2
  exit 2
fi

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "No project set. Run: gcloud config set project <PROJECT_ID>" >&2
  exit 2
fi

echo "==> deploying '$SERVICE' to $REGION in project $PROJECT"
cd "$ROOT"

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 40 \
  --timeout 3600 \
  --set-env-vars "HOST=0.0.0.0"

echo
echo "==> URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
