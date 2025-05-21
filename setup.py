
from setuptools import find_packages, setup

setup(
    name='predoc',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'predoc = predoc.trainer:main',
        ],
    },
    version='0.1.0',
    license='MIT'
)
