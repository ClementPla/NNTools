"""Setup file for the project.

This script is based on the setup.py script from imgaug (https://github.com/aleju/imgaug/blob/master/setup.py) to allow
for cases where a flavor of OpenCV is already installed. In such cases, that flavor is used. If no flavor is installed,
the headless flavor is installed (opencv-python-headless).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List

import re

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup


def check_alternative_installation(main: str, alternative_dependencies: List[str]) -> str:
    """Check if some alternative is already installed. If yes, return alternative, else return main."""
    for alternative in alternative_dependencies:
        try:
            alternative_pkg_name = re.split(r"[!<>=]", alternative)[0]
            get_distribution(alternative_pkg_name)
            return str(alternative)
        except DistributionNotFound:
            continue

    return str(main)


def get_install_requirements(main_dependencies: List[str], alternative_dependencies: Dict[str, List[str]]) -> List[str]:
    """Return a list of dependencies for installation. If some alternative is already installed, return alternative,
    else return main.

    Args:
        main_dependencies: list of main dependencies.
        alternative_dependencies: dict of alternative dependencies, where key is main dependency and value is list of
            alternative dependencies.

    Returns:
        list of dependencies for installation.
    """
    install_requires = []
    for dependency in main_dependencies:
        if dependency in alternative_dependencies:
            dependency = check_alternative_installation(dependency, alternative_dependencies.get(dependency))
        install_requires.append(dependency)

    return install_requires


DEPENDENCIES = [
    "torch",
    "numpy",
    "matplotlib",
    "tqdm",
    "pyyaml",
    "pandas",
    "torchvision",
    "opencv-python-headless",
    "segmentation_models_pytorch",
    "bokeh",
]

ALTERNATIVES = {
    "opencv-python-headless": ["opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"],
}

OPTIONAL_DEPENDENCIES = {
    "dev": ["black[jupyter]", "ruff"],
    "notebooks": ["ipykernel"],
    "tracker": ["mlflow"],
}


setup(
    install_requires=get_install_requirements(DEPENDENCIES, ALTERNATIVES),
    extras_require=OPTIONAL_DEPENDENCIES,
)
