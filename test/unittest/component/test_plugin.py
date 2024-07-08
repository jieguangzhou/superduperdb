import os
import shutil

import pytest

from superduperdb.components.plugin import Plugin

PACKAGE_PATH = "test/material/plugins/p_package"
MODULE_PATH = "test/material/plugins/p_package/p_model.py"
DIRECTORY_PATH = "test/material/plugins/p_directory"


def create_plugin(path, tempdirname):
    plugin_path = os.path.join(tempdirname, os.path.basename(path))
    if os.path.isfile(path):
        shutil.copy(path, plugin_path)
    else:
        shutil.copytree(path, plugin_path)

    return Plugin(path=plugin_path)


def test_package(tmpdir):
    with pytest.raises(ImportError):
        import p_package

        print(p_package)

    create_plugin(PACKAGE_PATH, tmpdir)
    from p_package.p_model import PModel  # ignore

    model = PModel("test")
    assert model.predict(1) == 2


def test_module(tmpdir):
    with pytest.raises(ImportError):
        import p_model

        print(p_model)

    create_plugin(MODULE_PATH, tmpdir)
    from p_model import PModel

    model = PModel("test")
    assert model.predict(1) == 2


def test_directory(tmpdir):
    with pytest.raises(ImportError):
        import p_directory

        print(p_directory)

    create_plugin(DIRECTORY_PATH, tmpdir)
    from p_directory.p_model_a import PModelA
    from p_directory.p_model_b import PModelB

    model_a = PModelA("test")
    model_b = PModelB("test")

    assert model_a.predict(1) == 2
    assert model_b.predict(1) == 2


def test_import(tmpdir):
    plugin_path = "test/material/data/exported_plugin"
    Plugin.read(plugin_path)

    with pytest.raises(ImportError):
        import p_model

        print(p_model)

    create_plugin(MODULE_PATH, tmpdir)
    from p_model import PModel

    model = PModel("test")
    assert model.predict(1) == 2
