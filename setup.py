import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xagg",
    version="0.1.2",
    author="Kevin Schwarzwald",
    author_email="kschwarzwald@iri.columbia.edu",
    description="Aggregating raster data over polygons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ks905383/xagg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
