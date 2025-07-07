"""
hizt: packed (compressed) histograms.
"""

from __future__ import annotations

__author__ = "Benjamin Fischer, Peter Fackeldey"
__copyright__ = "Copyright 2024, Benjamin Fischer & Peter Fackeldey"
__contact__ = "https://github.com/pfackeldey/hizt"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "0.0.1"


# expose public API
__all__ = [
    "Histogram",
    "__version__",
    "array_factory",
]


def __dir__():
    return __all__


from hizt import array_factory  # noqa: E402
from hizt.hist import Histogram  # noqa: E402
