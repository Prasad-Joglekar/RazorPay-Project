#!/usr/bin/env bash
# Publish the console to a Hugging Face **Static** Space.
#
# Static Spaces are the free tier -- Docker Spaces now need a paid plan. The
# page is self-contained (all data embedded, no network calls), so static
# hosting runs it exactly as it runs anywhere else. The one thing it cannot do
# is the live SSE console, which needs a Python process; see deploy/cloudrun/.
set -euo pipefail

SPACE_URL="${1:-}"
if [ -z "$SPACE_URL" ]; then
  echo "usage: $0 https://huggingface.co/spaces/<user>/<space>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning $SPACE_URL"
git clone --quiet "$SPACE_URL" "$WORK/space"
cd "$WORK/space"

# Spaces builds from main; a new Space clones empty and git would otherwise
# commit onto init.defaultBranch, which simply never deploys.
if git show-ref --verify --quiet refs/remotes/origin/main; then
  git checkout -q -B main origin/main
else
  git checkout -q -B main
fi

echo "==> staging index.html and README.md"
rm -f index.html README.md .gitattributes
cp "$HERE/index.html" index.html
cp "$HERE/README.md" README.md
cp "$ROOT/.gitattributes" .gitattributes

git add -A
if git diff --cached --quiet; then
  echo "==> Space already matches this repo; nothing to push"
  exit 0
fi

git commit --quiet -m "Publish fraud spike console"
echo "==> pushing to main (username + a write-scoped access token as the password)"
git push -u origin main
echo "==> done. It goes live within a few seconds at ${SPACE_URL%.git}"
