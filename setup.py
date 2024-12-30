import setuptools

package_name = "wenet_infer"

setuptools.setup(
    name=package_name,
    version='0.1.0',
    author="Meiluosi",
    author_email="meuluosi@gmail.com",
    package_dir={
        package_name: "py",
    },
    packages=[package_name],
    zip_safe=False,
    setup_requires=["tqdm"],
    install_requires=[
        "numpy",
        "requests",
        "tqdm",
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "librosa",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
    python_requires=">=3.8",
)
