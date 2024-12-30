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


setup(
    name="wenet_infer",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wenet_infer = py.wenet_model:load_model",
        ]
    },
)
