"""Fail fast on known Ray+vLLM worker dependency breakage.

Run inside the image after install (Dockerfile) or on a live pod:
  python /opt/ray-hive/smoke_test_worker_deps.py
"""
import importlib

# Import chains that have already bitten us at Serve inspect / model load time.
# Skip bare `vllm` on CPU builds — pin-memory warning triggers a circular import.
IMPORTS = [
    "numpy",
    "google.protobuf",
    "OpenSSL",
    "boto3",
    "botocore",
    "transformers",
    "ray.serve",
]

for name in IMPORTS:
    importlib.import_module(name)
    print(f"ok {name}")

import numpy
import torch
from OpenSSL import SSL, crypto  # noqa: F401 — GEN_EMAIL skew dies here
import google.protobuf as pb

np_parts = [int(x) for x in numpy.__version__.split(".")[:2]]
assert np_parts < [2, 4], f"numpy {numpy.__version__} needs X86_V2"

pb_major = int(pb.__version__.split(".", 1)[0])
assert pb_major == 5, f"protobuf {pb.__version__} (want 5.x: vLLM>=5.29.6, Ray breaks on 7.x)"

if torch.cuda.is_available():
    importlib.import_module("vllm")
    print("ok vllm")
    importlib.import_module("vllm.model_executor.models.qwen3_5")
    print("ok vllm.model_executor.models.qwen3_5")
else:
    print("skip vllm imports (no CUDA — expected during docker build)")

print("smoke ok")
