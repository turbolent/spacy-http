#!/usr/bin/env python

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='spacy-http',
      version='0.1.1',
      description='spaCy as a HTTP service',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords=["natural language processing", "nlp"],
      url='https://github.com/turbolent/spacy-http',
      author='Bastian Mueller',
      author_email='bastian@turbolent.com',
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Topic :: Text Processing",
      ],
      packages=['spacyHttp'],
      install_requires=[
          "click==6.7",
          "bottle==0.12.13",
          "meinheld==0.6.1",
          "spacy==2.0.12",
          "coloredlogs==10.0"
      ])
