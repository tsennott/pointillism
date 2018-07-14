import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pointillism",
    version="0.1.5",
    author="Tim Sennott",
    author_email="timothy.sennott@gmail.com",
    description="Pointillism-style photo manipulation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsennott/pointillism",
    packages=setuptools.find_packages(),
    install_requires=[
        "future",
        "numpy",
        "Pillow",
        "IPython; python_version>='3.0'",
        "IPython==5.4.0; python_version <'3.0'",  # LTS support for Python 2.7
        "matplotlib",
        "imageio",
        "scipy",
    ],
    test_suite='nose.collector',
    tests_require=['nose', 'mock'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='image manipulation art photos pointillism illustration graphics',
)
