#!/usr/bin/env python

import setuptools
import os

if __name__ == "__main__":

    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = current_script_dir

    with open(os.path.join(current_script_dir, "README.md"), "r") as f:
        long_description = f.read()

    with open(os.path.join(current_script_dir, "requirements.txt"), "r") as f:
        requirements = f.read().splitlines()

    packages = setuptools.find_packages(package_dir, exclude=("tests",))

    setuptools.setup(
        name="neurone_tools",
        version_format="{tag}",
        url="https://github.com/jkschluesener/pyneurone_tools",
        author="Jan K. Schluesener",
        author_email="code@jkschluesener.xyz",
        description="Binary file reader for NeurOne Tesla EEG data",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="GPL",
        include_package_data=True,
        packages=packages,
        install_requires=requirements,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
    )
