from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'Beneficial Computer-Vision Utilities'
LONG_DESCRIPTION = 'This repository is written in order to help the maker when doing Computer Vision projects'

# Setting up
setup(
    name='cv_utils',
    version=VERSION,
    author='Fadillah Adamsyah Maani',
    author_email='fadillahadam11@gmail.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    url='https://github.com/fadamsyah/cv_utils',
    keywords=['python', 'pytorch', 'computer vision', 'deep learning',
              'whole-slide images'],
    classifiers=[
        "Intended Audience :: Any",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)