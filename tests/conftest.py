import os
import tempfile

import pytest


@pytest.fixture
def temp_h5():
    fname = os.path.join(tempfile.gettempdir(), 'temp.h5')
    yield fname
    if os.path.isfile(fname):
        os.remove(fname)
