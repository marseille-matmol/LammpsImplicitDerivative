from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lammps_implicit_der',
    version='0.1',
    author="TD Swinburne, I Maliyov",
    author_email="thomas.swinburne@cnrs.fr",
    description="Implicit derivative for molecular statics with LAMMPS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.9',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'PyYAML',
        'matplotlib',
        'scipy',
        'psutil',
        'pynvml',
        'tqdm'
    ],
)

