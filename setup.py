import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywinEA", 
    version="0.5.0",
    author="Fernando Garcia",
    author_email="fernando.garciagu@alumnos.upm.es",
    packages=setuptools.find_packages(),
    license='MIT',
    description="Package with basic implementations of mono and multi-objective genetic algorithms for feature selection.",
    long_description=long_description,
    install_requires=['numpy', 'sklearn', 'tqdm', 'pandas', 'matplotlib'],
    long_description_content_type="text/markdown",
    url="https://github.com/FernandoGaGu/pywinEA",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)