import importlib.util
import os
import shutil
import sys
import typing as t

from superduperdb import Component, logging
from superduperdb.components.datatype import LazyFile, file_lazy


class Plugin(Component):
    """Plugin component allows to install and use external python packages as plugins.

    :param path: Path to the plugin package or module.
    :param identifier: Unique identifier for the plugin.
    :param cache_path: Path to the cache directory where the plugin will be stored.
    """

    type_id: t.ClassVar[str] = "plugin"
    _artifacts: t.ClassVar = (("path", file_lazy),)
    path: str
    identifier: str = ""
    cache_path: str = ".superduperdb/plugins"

    def __post_init__(self, db, artifacts):
        if isinstance(self.path, LazyFile):
            self._prepare_plugin()
        else:
            path_name = os.path.basename(self.path.rstrip("/"))
            self.identifier = self.identifier or f"plugin-{path_name}"
        self._install()
        super().__post_init__(db, artifacts)

    def _install(self):
        logging.debug(f"Installing plugin {self.identifier}")
        package_path = self.path
        path_name = os.path.basename(self.path.rstrip("/"))
        if os.path.isdir(package_path):
            init_file = os.path.join(package_path, "__init__.py")
            if "__init__.py" not in os.listdir(package_path):
                logging.info(f"Creating __init__.py file in {package_path}")
                open(init_file, "a").close()

            logging.debug(f"Plugin {self.identifier} is a package")
            spec = importlib.util.spec_from_file_location(
                path_name, os.path.join(package_path, "__init__.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[spec.name] = module
        else:
            if package_path.endswith(".py"):
                logging.debug(f"Plugin {self.identifier} is a standalone Python file")
                spec = importlib.util.spec_from_file_location(path_name, package_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[spec.name[:-3]] = module
            else:
                logging.error(
                    (
                        f"Plugin {self.identifier} path "
                        "is not a valid Python file or directory"
                    )
                )

    def _prepare_plugin(self):
        plugin_name_tag = f"{self.identifier}"
        assert isinstance(self.path, LazyFile)
        uuid_path = os.path.join(self.cache_path, self.uuid)
        # Check if plugin is already in cache
        if os.path.exists(uuid_path):
            names = os.listdir(uuid_path)
            assert len(names) == 1, f"Multiple plugins found in {uuid_path}"
            self.path = os.path.join(uuid_path, names[0])
            return

        logging.info(f"Preparing plugin {plugin_name_tag}")
        self.path = self.path.unpack()
        assert os.path.exists(
            self.path
        ), f"Plugin {plugin_name_tag} not found at {self.path}"

        # Pull the plugin to cache
        logging.info(f"Downloading plugin {self.identifier} to {self.path}")
        dist = os.path.join(self.cache_path, self.uuid, os.path.basename(self.path))
        if os.path.exists(dist):
            logging.info(f"Plugin {self.identifier} already exists in cache : {dist}")
        else:
            logging.info(f"Copying plugin [{self.identifier}] to {dist}")
            shutil.copytree(self.path, dist)

        self.path = dist
