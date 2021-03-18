from setuptools import setup, find_packages
import versioneer

setup(
    name="hierarchical_nu",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
