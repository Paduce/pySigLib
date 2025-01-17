from setuptools import find_packages, setup, Extension
from setuptools.command.install import install
from pathlib import Path
import shutil
import os
import platform

GETDLL = False

SYSTEM = platform.system()
DATA = ['libcpsig.dylib'] if SYSTEM == "Darwin" else ['cpsig.dll', 'cusig.dll']

class CustomInstall(install):
    def run(self):
        if GETDLL:
            parent_dir = Path(__file__).parent
            dir_ = parent_dir / 'pysiglib'

            if SYSTEM == "Windows":
                cusig_dll_path = parent_dir / 'siglib' / 'x64' / 'Release' / 'cusig.dll'
                cpsig_dll_path = parent_dir / 'siglib' / 'x64' / 'Release' / 'cpsig.dll'
                shutil.copy(cpsig_dll_path, dir_)
                shutil.copy(cusig_dll_path, dir_)
            elif SYSTEM == "Darwin":
                cpsig_dll_path = parent_dir / 'siglib' / 'x64' / 'Release' / 'libcpsig.dylib'
                shutil.copy(cpsig_dll_path, dir_)
            else:
                raise Exception("Unsupported OS during setup.py")

        super().run()

setup(
    name='pysiglib',
    version="1.0",
    packages=['pysiglib'],
    include_package_data=True,
    package_data={'': DATA},
    cmdclass={
        'install': CustomInstall,
    },
)
