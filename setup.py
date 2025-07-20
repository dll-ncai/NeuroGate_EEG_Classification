from setuptools import setup, find_packages

setup(
    name="neurogate_eeg",
    version="0.1.0",
    description="Run and evaluate the NeuroGATE EEG model.",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "torch", "mne", "scikit-learn", "tqdm", "matplotlib"
    ],
)

