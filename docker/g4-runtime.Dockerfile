ARG BASE_IMAGE=foundationpose-service:blackwell
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 \
        libgl1-mesa-glx \
        libegl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/g4-runtime-requirements.txt /tmp/g4-runtime-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/g4-runtime-requirements.txt

# RoboTwin v2.0 requires two upstream compatibility fixes that its installer
# normally applies in-place.  Keep them explicit and fail if upstream layout drifts.
RUN python3 - <<'PY'
from pathlib import Path

import mplib
import sapien

urdf_loader = Path(sapien.__file__).parent / "wrapper" / "urdf_loader.py"
text = urdf_loader.read_text()
old = 'srdf_file = urdf_file[:-4] + "srdf"'
new = 'srdf_file = urdf_file[:-4] + ".srdf"'
if old not in text and new not in text:
    raise RuntimeError(f"unexpected SAPIEN URDF loader layout: {urdf_loader}")
urdf_loader.write_text(text.replace(old, new))

planner = Path(mplib.__file__).parent / "planner.py"
text = planner.read_text()
old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
if old not in text and new not in text:
    raise RuntimeError(f"unexpected MPLib planner layout: {planner}")
planner.write_text(text.replace(old, new))
PY

COPY envs/curobo /opt/curobo
RUN python3 -m pip install --no-cache-dir --no-build-isolation /opt/curobo \
    && python3 -m pip install --no-cache-dir setuptools==69.5.1 warp-lang==1.12.0

ENV HOME=/tmp/g4-runtime \
    XDG_CACHE_HOME=/tmp/g4-runtime/.cache \
    MPLCONFIGDIR=/tmp/g4-runtime/matplotlib

WORKDIR /workspace/RoboTwin
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
