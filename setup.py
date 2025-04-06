from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-state-manipulator",
    version="0.1.0",
    author="Harish Santhana Lakshmi Ganesan",
    author_email="harishsg99@gmail.com",
    description="A tool for manipulating the internal neural activations of language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harishsg993010/neural-state-manipulator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.50.3",
        "numpy>=2.0.2",
        "bitsandbytes>=0.45.4",
    ],
)
