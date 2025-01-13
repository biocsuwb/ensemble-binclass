from setuptools import setup

setup(
    name="ensemble-binclass",
    version="1.0.6",
    author="Szymon Kolodziejski",
    author_email="koodziejskisz@outlook.com",
    description="Feature selection and ensemble classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/biocsuwb/ensemble-binclass",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9',
    install_requires=[
        "pandas~=2.2.0",
        "scikit-learn~=1.5.0",
        "xgboost~=2.0.0",
        "numpy~=2.2.0",
        "ReliefF~=0.1.2",
        "scipy~=1.15.0",
        "mrmr-selection~=0.2.8",
        "matplotlib~=3.10.0",
        "pytest~=8.3.0",
        "pytest-cov~=6.0.0",
        "seaborn~=0.13.0",
        "gprofiler-official~=0.3.5",
        "pypandoc~=1.15.0",
    ],
)
