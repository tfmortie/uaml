import os

version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_txt) as f:
    __version__ = f.read().strip()
