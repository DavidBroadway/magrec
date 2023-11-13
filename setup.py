import setuptools
import pathlib
import site
import sys

# odd bug with develop (editable) installs, see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

required = [
    "setuptools-git",  # see link at bottom: allows to specify package data to keep/upload/etc.
    "numpy",
    "matplotlib>=3.8.1",
    "matplotlib-scalebar>=0.8",
    "simplejson", 
    "pandas"
]

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

if __name__ == "__main__":
    setuptools.setup(
        name="magrec",
        version="0.1",
        author="David Broadway",
        author_email="broadwayphysics@gmail.com",
        description="Magnetisation reconstruction software.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/DavidBroadway/magrec",
        keywords=[
            "NV",
            "QDM",
            "Diamond",
            "Quantum",
            "Quantum Sensing",
        ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Development Status :: 2 - Pre-Alpha",
        ],
        license="MIT",
        package_dir={"": "src"},
        packages=setuptools.find_packages(
            where="", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
        ),
        install_requires=required,
        # python_requires=">=3.10",  # check pyfftw
        package_data={"": ["*.md", "*.json"]},
        setup_requires=["wheel"],  # force install of wheel first? Untested 2021-08-01.
    )

