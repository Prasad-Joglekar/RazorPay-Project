# Container for the live streaming console (`razorpay_fraud live`).
#
# No pip install step, on purpose. The detector, simulator, streaming layer and
# the console's HTTP server are pure standard library; scikit-learn and
# matplotlib are only needed by `demo`, which builds the comparison models and
# the charts and does not run here. Skipping them keeps the image around 150 MB
# instead of roughly 1 GB and makes the build near-instant -- which matters on a
# free tier, where a slow build is the difference between a judge seeing the
# demo and giving up.
FROM python:3.12-slim

# Hugging Face Spaces runs containers as UID 1000 and will not write as root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user:user razorpay_fraud/ ./razorpay_fraud/

# Spaces routes traffic to app_port from README.md and requires binding all
# interfaces. cli.py reads both from the environment, so no code change here.
ENV HOST=0.0.0.0 \
    PORT=7860
EXPOSE 7860

# --days 1 gives a ~6s cold start and ~126 MB peak, comfortably inside a free
# tier. The default 3 days costs ~14s and ~170 MB; raise it if the Space has
# room and you want a longer stream to replay.
CMD ["python", "-m", "razorpay_fraud", "live", "--days", "1", "--speed", "300"]
