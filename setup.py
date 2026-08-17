from setuptools import find_packages, setup

setup(
    name="agentic-rag-tool-routing",
    version="0.1.0",
    description="Agentic RAG (dense retrieval + cross-encoder reranking), semantic tool routing, QLoRA/PEFT, and LLM-serving optimizations (KV-cache reuse, speculative decoding) -- all trained and benchmarked from scratch, offline.",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=["torch>=2.1", "numpy>=1.24", "scipy>=1.10", "pyyaml>=6.0"],
)
