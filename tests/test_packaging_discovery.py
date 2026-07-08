from setuptools import find_packages


def test_setuptools_discovers_latticetn_subpackages():
    packages = set(find_packages(include=["latticetn*"]))
    assert "latticetn" in packages
    assert "latticetn.benchmarks" in packages
