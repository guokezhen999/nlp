from setuptools import setup, find_packages

setup(
    name='mydl',
    version='0.1.0',
    packages=find_packages(include=['mydl', 'mydl.*']),
    install_requires=[
        "torch", "numpy", "matplotlib", "IPython"
    ],
)