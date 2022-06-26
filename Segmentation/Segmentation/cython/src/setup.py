# import numpy
# from setuptools import setup
# from Cython.Build import cythonize
# setup(
#     ext_modules=cythonize("lpbox.pyx"),
#     include_dirs=numpy.get_include(),
# )

import numpy 
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("lpbox",["lpbox.pyx"], language="c++", 
                libraries=['opencv_shape', 'opencv_stitching', 'opencv_superres', 'opencv_videostab', 'opencv_aruco', 'opencv_bgsegm', 'opencv_bioinspired', 'opencv_ccalib', 'opencv_datasets', 'opencv_dpm', 'opencv_face', 'opencv_freetype', 'opencv_fuzzy', 'opencv_hdf', 'opencv_line_descriptor', 'opencv_optflow', 'opencv_video', 'opencv_plot', 'opencv_reg', 'opencv_saliency', 'opencv_stereo', 'opencv_structured_light', 'opencv_phase_unwrapping', 'opencv_rgbd', 'opencv_viz', 'opencv_surface_matching', 'opencv_text', 'opencv_ximgproc', 'opencv_calib3d', 'opencv_features2d', 'opencv_flann', 'opencv_xobjdetect', 'opencv_objdetect', 'opencv_ml', 'opencv_xphoto', 'opencv_highgui', 'opencv_videoio', 'opencv_imgcodecs', 'opencv_photo', 'opencv_imgproc', 'opencv_core'],
                library_dirs=['/usr/local/lib', '/usr/local/include/opencv2', '/usr/local/include/Eigen4']
    )]

setup(
    # name = "lpbox",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=numpy.get_include()
)