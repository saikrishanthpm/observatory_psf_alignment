from setuptools import setup, find_packages

setup(
    name='Batoid4LOFT',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'batoid',
        'matplotlib', 
        'plotly', 
        'pandas', 
        'scipy'
    ],
    description='Optics simulation and analysis package.',
)