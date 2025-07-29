from setuptools import setup, find_packages

setup(
    name="dlclus",
    version="0.1.0",
    description="Deep learning clustering tools for SBND",
    author="Haiwang Yu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "jupyter",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.6",
)
