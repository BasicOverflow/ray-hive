"""Setup configuration for ray-hive package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ray-hive",
    version="0.1.0",
    author="Your Name",
    description="Distributed LLM serving engine for Ray clusters",
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
        "ray[serve]==2.53.0",
        "vllm>=0.11.1",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "transformers>=4.40.0",
    ],
)

