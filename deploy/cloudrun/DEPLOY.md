# Deploying the live console to Google Cloud Run

Cloud Run's always-free tier covers this comfortably: 2 million requests,
180,000 vCPU-seconds and 360,000 GiB-seconds per month. The service scales to
zero, so it costs nothing while nobody is watching.

A card must be on file for the account. Nothing is charged inside the free
limits, but if that is a problem, use a Cloudflare quick tunnel from your laptop
instead — it needs no account at all.

## 1. Install and sign in

Install the SDK: <https://cloud.google.com/sdk/docs/install>

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Create a project first at <https://console.cloud.google.com/projectcreate> if
you do not have one. The `services enable` step is easy to forget and the
deploy fails with a permissions error that does not obviously say so.

## 2. Deploy

```bash
./deploy/cloudrun/deploy.sh
```

First deploy takes 3–5 minutes: Cloud Build builds the image, pushes it to
Artifact Registry, then Cloud Run starts it. Subsequent deploys are faster.

The script prints the service URL when it finishes. Override the defaults with
environment variables if you want:

```bash
SERVICE=my-console REGION=europe-west1 ./deploy/cloudrun/deploy.sh
```

## 3. Check it

Open the URL. You should see a warm-up progress bar for a few seconds, then the
LIVE pill turns red, the IST clock ticks, and alerts start arriving.

The first visit after an idle period is slow — the container has to start and
then warm detector state. That is scale-to-zero working as intended, not a bug.

## Why these flags

| flag | reason |
|---|---|
| `--max-instances 1` | **The important one.** Engine state lives in the process. With more than one instance, two viewers would see different points in the stream, and Pause/Restart would affect only whichever instance served that click. One instance means one shared stream, which is what the console is designed around. |
| `--timeout 3600` | An SSE connection is a single long-lived request. The 300s default would cut the stream every five minutes. The page reconnects and is handed a fresh snapshot, so it recovers — but it visibly hitches. |
| `--allow-unauthenticated` | Reviewers must be able to open it without a Google account. |
| `--memory 512Mi` | Measured peak is ~126 MB at `--days 1`. 512Mi is the smallest comfortable size. |
| `--concurrency 40` | Each open SSE connection holds a thread for as long as the browser stays. 40 is far more viewers than this will ever have and keeps thread count sane. |
| `--min-instances 0` | Scales to zero, which is what keeps it free. The cost is a cold start. |
| `--port 8080` | Cloud Run injects `PORT`; the Dockerfile and CLI both default to 8080 so the two agree. |

## CPU allocation, and why the design happens to fit

By default Cloud Run only allocates CPU while a request is in flight. The
detector runs on a background thread, which would normally be starved between
requests — but an open SSE connection *is* an in-flight request, so the CPU
stays allocated for exactly as long as somebody is watching, and the stream
pauses when nobody is. That is the behaviour you want, and it costs nothing
extra.

If you ever want the stream to advance with no viewers connected, that needs
`--no-cpu-throttling`, which is billed outside the free tier.

## Known limits

- **The data is simulated, and ground-truth labels are sent to the browser** so
  the page can show live precision. That makes it a demonstration, not a
  production dashboard. Worth saying so on your submission.
- **One shared stream.** Everyone sees the same position, and the Pause,
  Restart and speed controls affect every viewer. Right for a demo, wrong for a
  tool.
- **Cold starts.** If reviewers are looking at a known time, open the URL
  yourself a minute beforehand.

## Tearing it down

```bash
gcloud run services delete fraud-console --region asia-south1
```
