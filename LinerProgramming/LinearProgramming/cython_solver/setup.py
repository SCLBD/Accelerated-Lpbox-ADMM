# from distutils.core import setup, Extension
# from Cython.Build import cythonize
import eigency


# extensions = [
#     Extension("lpbox",
#                ["lpbox.pyx"],
#                include_dirs=[".", '/usr/local/include/Eigen4']+eigency.get_includes(include_eigen=False) 
#                 )

# ]

# setup(
#     name='lpbox',
#     version='1.0',
#     ext_modules=cythonize(extensions)
# )

import numpy
from setuptools import setup
from Cython.Build import cythonize
setup(
    ext_modules=cythonize("lpbox.pyx"),
    include_dirs=numpy.get_include(),
)