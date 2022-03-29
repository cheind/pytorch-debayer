from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent

with open("requirements.txt") as f:
    required = f.read().splitlines()

main_ns = {}
with open(THISDIR / "debayer" / "__version__.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="pytorch-debayer",
    version=main_ns["__version__"],
    description="Convolutional Debayering/Demosaicing layer for PyTorch",
    author="Christoph Heindl",
    url="https://github.com/cheind/pytorch-debayer/",
    license="MIT",
    install_requires=required,
    packages=find_packages(".", include="debayer*"),
    include_package_data=True,
    keywords="debayer bayer pytorch convolution",
)
