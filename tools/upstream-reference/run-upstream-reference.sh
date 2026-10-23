#!/bin/sh
# Prepare the upstream reference environment inside the pod and run the harness.
set -eu
COMMIT=02561132682c2a66925bb0df9f6a1d3f9e816723
W=/work
mkdir -p $W/pylib $W/src $W/tmp
python3 -m pip install --quiet --no-cache-dir --disable-pip-version-check --target $W/pylib \
  transformers==4.57.1 triton==3.2.0 einops==0.8.1 >/dev/null
python3 - <<PY
import io, tarfile, urllib.request
data = urllib.request.urlopen("https://codeload.github.com/BICLab/SpikingBrain2.0/tar.gz/$COMMIT", timeout=120).read()
tarfile.open(fileobj=io.BytesIO(data)).extractall("$W/src", filter="data")
PY
ROOTDIR=$(ls -d $W/src/SpikingBrain2.0-*)
mkdir -p $W/shim/flash_attn && cp /tools/flash_attn_shim.py $W/shim/flash_attn/__init__.py && cp /tools/flash_attn_interface_stub.py $W/shim/flash_attn/flash_attn_interface.py
export PYTHONPATH=$W/shim:$ROOTDIR:$ROOTDIR/flash-linear-attention_dev:$ROOTDIR/MoBA:$W/pylib:/opt/spikingbrain
export TRITON_INTERPRET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1
python3 -c "import torch, triton, transformers; print('versions', torch.__version__, triton.__version__, transformers.__version__)"
exec python3 /tools/upstream_reference.py /models/spikingbrain /work/upstream-reference.jsonl "${DECODE_STEPS:-4}"
