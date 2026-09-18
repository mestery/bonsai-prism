"""Package wrapper for the official Prism hadamard runtime.

The vendored runtime modules (``runtime.py``, ``artifact.py``, ``codec.py``,
``vision_artifact.py``) are taken verbatim from the official Hugging Face
repository and use flat sibling imports such as ``from codec import transcode``.
That style only resolves if this directory itself is on ``sys.path``.

We therefore add the directory to ``sys.path`` here so that ``import runtime``
(and friends) resolve to the top-level module files, while the package remains
importable as ``runtime.runtime``.
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
