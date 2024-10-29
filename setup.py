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

"""
DEPRECATION: Legacy editable install of ml-accelerator==1.0.0 from 
file:///Users/simongarciamorillo/Library/CloudStorage/OneDrive-Personal/Documents/BetterTradeGroup/ml-accelerator (setup.py develop) is deprecated. 
pip 25.0 will enforce this behaviour change. A possible replacement is to add a pyproject.toml or enable --use-pep517, and use setuptools >= 64. 
If the resulting installation is not behaving as expected, try using --config-settings editable_mode=compat. 
Please consult the setuptools documentation for more information. Discussion can be found at https://github.com/pypa/pip/issues/11457
"""