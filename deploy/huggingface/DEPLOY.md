# Deploying the live console to Hugging Face Spaces

> **Docker Spaces now require a paid Hugging Face plan** -- only Static Spaces
> are free. For a free live deployment see `deploy/cloudrun/` instead; these
> instructions still apply if you have a PRO subscription.

The Space runs `python -m razorpay_fraud live` in a Docker container. the image needs no third-party packages, and the app peaks around
126 MB with a ~6 second cold start.

## 1. Create the Space

At <https://huggingface.co/new-space>:

| field | value |
|---|---|
| Owner / Space name | your account, e.g. `fraud-spike-console` |
| License | your choice |
| **Select the SDK** | **Docker** → **Blank** |
| Hardware | CPU basic (free) |
| Visibility | **Public** — judges must be able to open it without an account |

## 2. Push the code

A Space is a git repo. It needs three things at its root: `Dockerfile`,
`README.md` **with the Hugging Face frontmatter**, and `razorpay_fraud/`.

The helper script assembles exactly that and pushes it:

```bash
./deploy/huggingface/push-space.sh https://huggingface.co/spaces/<user>/<space>
```

Git will ask for credentials on push: use your Hugging Face **username**, and an
access token with **write** scope as the password (create one at
<https://huggingface.co/settings/tokens>). Your account password will not work.

<details>
<summary>Doing it by hand instead</summary>

```bash
git clone https://huggingface.co/spaces/<user>/<space> space
cp -r razorpay_fraud Dockerfile space/
cp deploy/huggingface/README.md space/README.md   # note: renamed to README.md
cd space && git add -A && git commit -m "Deploy live fraud console" && git push
```

The rename matters. Spaces reads `sdk: docker` and `app_port: 8080` from the
frontmatter of the repo's own `README.md`; without it the Space will not start.
</details>

## 3. Watch the build

The Space page shows **Building** then **Running**. First build takes a couple
of minutes, almost all of it pulling `python:3.12-slim`.

When it opens, the page shows a warm-up progress bar for a few seconds — the
detector is building sliding-window state from earlier traffic before it scores
anything — then switches to live streaming.

## Configuration

Change these in the `CMD` line of the `Dockerfile` and push again:

| flag | default here | effect |
|---|---|---|
| `--days` | `1` | Size of the simulated stream. `3` is the project default: ~14s cold start, ~170 MB. |
| `--speed` | `300` | Stream speed. 300× means one stream-hour every 12 seconds. |
| `--threshold` | (cost-optimal) | Alert threshold. Lower alerts more. |

`HOST` and `PORT` are read from the environment and are already set correctly by
the `Dockerfile`; don't hard-code them.

## Known limits

- **The data is simulated, and ground-truth labels are sent to the browser** so
  the page can show live precision. That is what makes it a demonstration, not
  a production dashboard. Worth stating on your submission so it reads as a
  deliberate choice rather than an oversight.
- **One thread per connected browser.** `ThreadingHTTPServer` holds a thread for
  each open SSE connection. Fine for judges; it would not survive real traffic.
- **Free Spaces sleep after inactivity.** A cold visitor waits for the container
  to wake plus the ~6s warm-up. If that matters for judging, either upgrade the
  hardware or open the Space yourself shortly before the review window.
- **State is per-process, not per-visitor.** Everyone watching sees the same
  stream at the same position, and the Pause / Restart / speed controls affect
  every viewer. That is the right model for a demo and the wrong one for a tool.
