from setuptools import setup, find_packages

# Define requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

description = """
ml_accelerator is a propietary library designed to accelerate ML proyects.
"""

# Define setup
setup(
    name="ml_accelerator",
    description=description,
    author="Simón P. García Morillo",
    author_email="simongmorillo1@gmail.com",
    version="1.0.0",
    install_requires=requirements,
    packages=find_packages(),
    package_data={"test": ["test/*"]}, # "config": ["config/*"], 
    long_description=open("README.md").read(),
    license=open("LICENSE").read()
)