import platform
from setuptools import setup, find_packages

requirements = [
    "numpy",
    "requests",
    "tqdm",
    "torch>=1.13.0",
    "torchaudio>=0.13.0",
    "openai-whisper",
    "librosa",
]
if platform.system() == 'Windows':
    requirements += ['PySoundFile']

setup(
    name="wenet_encapsulation",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wenet_encapsulation=model:load_model",
    ]},
)
