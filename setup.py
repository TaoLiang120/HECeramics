#!/usr/bin/env python

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

with open('LICENSE.rst') as f:
    license = f.read()

setup(name='heceramics',
      version='1.0.0',
      description='a machine learning program to predict carbon vacancy formation energy in high entropy carbides',
      long_description=open(os.path.join(module_dir, 'README.rst')).read(),
      url='https://github.com/TaoLiang/ml4hec',
      author='Tao Liang',
      author_email='tliang7@utk.edu',
      license='MIT',
      packages=find_packages(),
      package_data={"heceramics.myelements": ["*.json", "*.csv"],
                    "heceramics": ["*.csv"]},
      entry_points={
          'console_scripts': ['run_HECs = heceramics.script.run:main']
      },
      install_requires=["numpy>=1.10.3",
                        "scipy>=0.17.1",
                        "matplotlib>=1.5.1",
                        "pymatgen>=4.4.0",
                        "scikit-learn==1.5.0",
                        "pandas>=0.20.3"],
      )
