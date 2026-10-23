# CPU-only inference server for SpikingBrain2.0 text models. Weights are not
# part of the image: mount the checkpoint directory at /models/spikingbrain.
ARG PYTHON_BASE=docker.io/library/python:3.12-slim
FROM ${PYTHON_BASE}
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt \
 && python3 -c "import torch; print(torch.__version__)"
COPY spikingbrain_cpu/ /opt/spikingbrain/spikingbrain_cpu/
COPY weights-manifest.json /opt/spikingbrain/weights-manifest.json
# OpenShift runs containers with an arbitrary non-root UID: keep code world-readable.
RUN chmod -R a+rX /opt/spikingbrain
ENV PYTHONPATH=/opt/spikingbrain PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    OMP_NUM_THREADS=4 SB_MODEL_DIR=/models/spikingbrain
USER 1001
WORKDIR /tmp
EXPOSE 8080
ENTRYPOINT ["python3", "-m", "spikingbrain_cpu.server"]
