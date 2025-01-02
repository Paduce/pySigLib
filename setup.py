from setuptools import find_packages, setup, Extension
from setuptools.command.install import install
from pathlib import Path
import shutil
import os

class CustomInstall(install):
    def run(self):
        parent_dir = Path(__file__).parent
        dir_ = parent_dir / 'pysiglib'

        cusig_dll_path = parent_dir / 'siglib' / 'x64' / 'Release' / 'cusig.dll'
        cpsig_dll_path = parent_dir / 'siglib' / 'x64' / 'Release' / 'cpsig.dll'

        shutil.copy(cpsig_dll_path, dir_)
        shutil.copy(cusig_dll_path, dir_)

setup(
    name='pysiglib',
    version="1.0",
    packages=['pysiglib'],
    include_package_data=True,
    package_data={'': ['cpsig.dll', 'cusig.dll']},
    cmdclass={
        'install': CustomInstall,  # Custom install command to modify PATH
    },
)
