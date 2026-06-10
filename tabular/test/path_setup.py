import importlib.util
import os
import sys


def ensure_dataprep_importable(test_file):
    current_dir = os.path.dirname(os.path.abspath(test_file))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    package_parent = os.path.abspath(os.path.join(project_root, ".."))

    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    if "dataprep" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        "dataprep",
        os.path.join(project_root, "__init__.py"),
        submodule_search_locations=[project_root],
    )
    dataprep_module = importlib.util.module_from_spec(spec)
    sys.modules["dataprep"] = dataprep_module
    spec.loader.exec_module(dataprep_module)
