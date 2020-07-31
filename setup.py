import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="oos_detect",
    version="0.0.1",
    author="dunkelhaus",
    author_email="jena.suraj.k@gmail.com",
    description="OOS detection for CLINC dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dunkelhaus/oos-detect",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
