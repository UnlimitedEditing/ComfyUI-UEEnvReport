"""EnvProbe: reports the Python / torch / GPU / network environment a Graydient job really runs in.

Stdlib only at import time, so it still loads (and prints its findings to the ComfyUI log) on a
half-configured machine. The node itself returns the report as a data image (UTF-8 JSON in the pixels).
"""
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

URLS = {
    "huggingface": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
    "github": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    "bbc_images": "https://ichef.bbci.co.uk/ace/standard/976/cpsprodpb/4b56/live/9be50560-cc9f-11f1-94e6-edd4c60b2e1d.jpg",
    "pypi": "https://pypi.org/simple/pip/",
}


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip()[:1500]
    except Exception as e:
        return f"unavailable: {e}"


def _head(url):
    t = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return {"status": r.status, "seconds": round(time.time() - t, 2)}
    except Exception as e:
        return {"error": str(e)[:200], "seconds": round(time.time() - t, 2)}


def collect(network=True):
    info = {
        "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
        "hostname": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
    }
    try:
        info["meminfo"] = [l.strip() for l in open("/proc/meminfo").read().splitlines()[:3]]
    except Exception:
        pass
    pkgs = {}
    try:
        from importlib import metadata
        for d in metadata.distributions():
            pkgs[d.metadata["Name"] or "?"] = d.version
    except Exception as e:
        pkgs["error"] = str(e)
    keep = ("torch", "torchvision", "torchaudio", "numpy", "pillow", "transformers", "tokenizers", "safetensors",
            "onnxruntime", "kokoro-onnx", "accelerate", "scipy", "av", "aiohttp", "einops", "comfy-aimdo",
            "comfy_kitchen", "comfyui-frontend-package", "sqlalchemy", "alembic", "triton")
    info["key_packages"] = {k: v for k, v in sorted(pkgs.items()) if k.lower().replace("_", "-") in keep}
    info["package_count"] = len(pkgs)
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                         "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["torch"]["device"] = torch.cuda.get_device_name(0)
            info["torch"]["capability"] = list(torch.cuda.get_device_capability(0))
            info["torch"]["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:
        info["torch"] = {"error": str(e)[:300]}
    info["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    info["ffmpeg"] = (shutil.which("ffmpeg") or "none")
    if info["ffmpeg"] != "none":
        info["ffmpeg_version"] = _run(["ffmpeg", "-version"]).splitlines()[0]
    info["df"] = _run(["df", "-h", "/", os.getcwd(), "/tmp"])
    if network:
        info["network"] = {k: _head(u) for k, u in URLS.items()}
    return info


# Written to the ComfyUI log at import time, before any node runs.
try:
    print("[EnvProbe] " + json.dumps(collect(network=False), default=str))
except Exception as e:  # never break startup
    print(f"[EnvProbe] import-time probe failed: {e}")


class UEEnvProbeReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"note": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("report_data",)
    FUNCTION = "run"
    CATEGORY = "EnvProbe"

    def run(self, note):
        import numpy as np
        import torch
        info = collect(network=True)
        info["note"] = note
        text = json.dumps(info, default=str)
        print("[EnvProbe] full " + text)
        raw = text.encode("utf-8")
        payload = len(raw).to_bytes(4, "big") + raw
        while len(payload) % 3:
            payload += b"\x00"
        side = max(1, math.ceil(math.sqrt(len(payload) // 3)))
        payload += b"\x00" * (side * side * 3 - len(payload))
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"UEEnvProbeReport": UEEnvProbeReport}
NODE_DISPLAY_NAME_MAPPINGS = {"UEEnvProbeReport": "UE Env Probe Report"}
