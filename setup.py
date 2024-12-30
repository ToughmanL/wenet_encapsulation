from setuptools import setup, find_packages

requirements = [
    "requests",
    "tqdm",
    "torch==2.2.2",
    "torchaudio==2.2.2",
    "openai-whisper",
    "librosa",
    "pyyaml",
    "langid"
]


setup(
    name="wenet_infer",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wenet_infer = wenet_infer.wenet_model:load_model",
        ]
    },
)
