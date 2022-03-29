from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent


def read_requirements(fname):
    with open(THISDIR / "requirements" / fname, "r") as f:
        return f.read().splitlines()


core_required = read_requirements("core.txt")
apps_required = read_requirements("apps.txt")
dev_required = read_requirements("dev.txt") + apps_required

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
    install_requires=core_required,
    packages=find_packages(".", include="debayer*"),
    include_package_data=True,
    keywords="debayer bayer pytorch convolution",
    extras_require={
        "dev": dev_required,
        "apps": apps_required,
        "full": core_required + dev_required,
    },
)
