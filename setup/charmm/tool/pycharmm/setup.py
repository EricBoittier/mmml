import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pycharmm',
    version='0.0.1',
    author='Josh Buckner',
    author_email='bucknerj@umich.edu',
    description='a python library for molecular dynamics with CHARMM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://brooks.chem.lsa.umich.edu',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU GPLv3 License',
        'Operating System :: OS Independent',
    ),
    python_requires='>=3.4',
    install_requires=['pandas', 'numpy', 'scipy'],
)
