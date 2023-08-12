from setuptools import setup

setup(
    name="torchns",
    version="0.0.1",
    description="Nested sampling in torch",
    url="https://github.com/undark-lab/torch-ns",
    author="Christoph Weniger, Noemi Anau Montel",
    author_email="c.weniger@uva.nl, n.anaumontel@uva.nl",
    packages=["torchns"],
    install_requires=[
        "numpy>=1.18.1",
        "torch>=1.10.2",
        "tqdm>=4.46.0",
        "matplotlib>=3.1.3",
        "corner>=2.2.1",
    ],
)
