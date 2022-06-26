from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

# setup(name='LinearProgramming',
#       py_modules=['LinearProgramming'],
#       install_requires=[
#           'torch'
#       ],
# )

setup(name='Segmentation',
      py_modules=['Segmentation'],
      install_requires=[
          'torch'
      ],
)

# setup(name='SparseAttack',
#       py_modules=['SparseAttack'],
#       install_requires=[
#           'torch'
#       ],
# )