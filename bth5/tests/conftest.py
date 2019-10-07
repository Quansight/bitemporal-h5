import h5py
import platform


def pytest_cmdline_preparse(args):
    version_tuple = h5py.version.version_tuple
    if version_tuple[:2] >= (2, 9) and platform.system == "Linux":
        args.append("--doctest-modules")
