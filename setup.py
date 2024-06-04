# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import setup, find_packages


NAME = 'audiocraft'
DESCRIPTION = 'Audio generation research library for PyTorch'

URL = 'https://github.com/facebookresearch/audiocraft'
AUTHOR = 'FAIR Speech & Audio'
EMAIL = 'defossez@meta.com, jadecopet@meta.com'
REQUIRES_PYTHON = '>=3.8.0'

for line in open('audiocraft/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author_email=EMAIL,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require={
        'dev': ['coverage', 'flake8', 'mypy', 'pdoc3', 'pytest'],
        'wm': ['audioseal'],
    },
    packages=[p for p in find_packages() if p.startswith('audiocraft')],
    package_data={'audiocraft': ['py.typed']},
    include_package_data=True,
    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
