"""Setup configuration for vram-scheduler package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vram-scheduler",
    version="0.1.0",
    author="Your Name",
    description="VRAM-aware LLM serving for Ray clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ray-k3s-deployment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ray[serve]>=2.8.0",
        "vllm>=0.2.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
    ],
)

