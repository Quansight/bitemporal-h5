import h5py
import platform


def pytest_cmdline_preparse(args):
    version = h5py.__version__.split(".")
    version_tuple = []
    for i in version:
        try:
            version_tuple.append(int(i))
        except ValueError:
            pass
    version_tuple = tuple(version_tuple)

    if version_tuple[:2] >= (2, 9) and platform.system == "Linux":
        args.append("--doctest-modules")
