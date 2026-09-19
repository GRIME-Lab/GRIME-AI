import sys
import importlib
import importlib.abc
import importlib.util

_ALIAS = "appcore"


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve 'appcore.x.y' to this package's real 'x.y' module (same object, no second copy)."""

    def __init__(self, alias, real):
        self.alias, self.real = alias, real

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self.alias + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(self.real + spec.name[len(self.alias):])

    def exec_module(self, module):
        pass


if _ALIAS not in sys.modules:
    sys.modules[_ALIAS] = sys.modules[__name__]
    sys.meta_path.insert(0, _AliasFinder(_ALIAS, __name__))

from .app_identity import USER_ROOT as PROJECT_ROOT