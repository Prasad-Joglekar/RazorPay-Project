#!/usr/bin/env bash
# Assemble a Hugging Face Space from this repo and push it.
#
# A Space needs three things at its root: Dockerfile, README.md carrying the
# Hugging Face frontmatter, and the razorpay_fraud package. This copies exactly
# those into a clone of the Space repo, so nothing else in the project leaks
# into the deployment and the rename to README.md cannot be forgotten.
set -euo pipefail

SPACE_URL="${1:-}"
if [ -z "$SPACE_URL" ]; then
  echo "usage: $0 https://huggingface.co/spaces/<user>/<space>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning $SPACE_URL"
git clone --quiet "$SPACE_URL" "$WORK/space"
cd "$WORK/space"

# Spaces builds from main. A freshly created Space clones empty, and git
# would otherwise commit onto whatever init.defaultBranch happens to be --
# push master to a Space and it simply never builds. When the Space already
# has history, branch from it, or the push is rejected as a non-fast-forward.
if git show-ref --verify --quiet refs/remotes/origin/main; then
  git checkout -q -B main origin/main
else
  git checkout -q -B main
fi

echo "==> staging Dockerfile, README.md and razorpay_fraud/"
rm -rf razorpay_fraud Dockerfile README.md .gitattributes
cp "$ROOT/Dockerfile" .
# Without this the Space checks files out with CRLF on Windows, so every run
# would look like a full-file change and push a pointless commit.
cp "$ROOT/.gitattributes" .
cp "$ROOT/deploy/huggingface/README.md" README.md
mkdir -p razorpay_fraud
# Source and the served page only -- no __pycache__, no tests, no out/.
for f in "$ROOT"/razorpay_fraud/*.py "$ROOT"/razorpay_fraud/*.html; do
  [ -e "$f" ] && cp "$f" razorpay_fraud/
done

git add -A
if git diff --cached --quiet; then
  echo "==> Space already matches this repo; nothing to push"
  exit 0
fi

git commit --quiet -m "Deploy live fraud console"
echo "==> pushing to main (username + a write-scoped access token as the password)"
git push -u origin main
echo "==> done. Watch the build at ${SPACE_URL%.git}"
