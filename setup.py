from setuptools import find_packages, setup

setup(
    name="summer_school_2025",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.8",
)
