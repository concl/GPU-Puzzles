from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'naive_cpp_matmul',
        ['src/matmul.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

setup(
    name='naive_cpp_matmul',
    version='0.1.0',
    description='Naive C++ Matrix Multiplication using PyBind11',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
