from setuptools import setup, find_packages

DESCRIPTION = 'Estimate Cn2 from Speckle patterns'
LONG_DESCRIPTION = 'A Python package that uses machine learning to predict turbolence profiles from static patterns'

# Setting up
setup(
    name='speckcn2',
    version='{{VERSION_PLACEHOLDER}}',
    author='Simone Ciarella',
    author_email='<s.ciarella@esciencecenter.com>',
    url='https://github.com/MALES-project/SpeckleCn2Profiler',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    python_requires='>=3.10.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'torchvision',
        'scipy',
        'PyYAML',
    ],
    keywords=[
        'python', 'optical satelites', 'machine learning', 'turbolence',
        'laser communication'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: Unix',
        'Topic :: Scientific/Engineering',
    ],
)
