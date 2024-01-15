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
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'torchvision',
        'scipy',
    ],
    keywords=[
        'python', 'optical satelites', 'machine learning', 'turbolence',
        'laser communication'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Education',
        'Programming Language :: Python :: 3',
        'Operating System :: Unix',
    ],
)
