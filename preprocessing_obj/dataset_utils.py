"""Locate prepared object datasets under the data root."""
import os


def discover_datasets(data_root):
    """Return object-dataset names under data_root (subdirs containing a `renders/`
    folder), e.g. "ABO" and "3D-Future"."""
    if not os.path.isdir(data_root):
        return []
    return [name for name in sorted(os.listdir(data_root))
            if os.path.isdir(os.path.join(data_root, name, "renders"))]
