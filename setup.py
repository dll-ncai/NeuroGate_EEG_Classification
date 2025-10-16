from setuptools import setup, find_packages

setup(
    name="cereprocess",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "torch", "mne", "scikit-learn", "tqdm", "matplotlib", "torchmetrics", "ipywidgets"
    ],
)

