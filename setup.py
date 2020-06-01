#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['PILasOPENCV']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Carl Asman",
    author_email='github@edlin.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Simple captchagenerator",
    install_requires=requirements,
    license="MIT license",
    long_description="",
    include_package_data=True,
    keywords='captchagenerator',
    name='captchagenerator',
    packages=find_packages(include=['captchagenerator']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/EdlinOrg/captchagenerator',
    version='0.0.1',
    zip_safe=False,
)
