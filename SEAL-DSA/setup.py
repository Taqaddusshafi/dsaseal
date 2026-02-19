from setuptools import setup, find_packages

setup(
    name="seal-dsa",
    version="1.0.0",
    description="SEAL: Simplified Self-Adapting Language Model for DSA Education using LoRA",
    author="Your Name",
    author_email="your.email@university.edu",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.16.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "seal-dsa=seal_dsa.main:main",
        ],
    },
)
