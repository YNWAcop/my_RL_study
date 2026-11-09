"""Configure the selected virtual environment for the Windows PARL course."""
from pathlib import Path
import sys
import sysconfig
import textwrap


def main():
    expected = Path(__file__).resolve().parents[1] / ".venv"
    if Path(sys.prefix).resolve() != expected:
        raise SystemExit("Run this script using this repository's .venv/Scripts/python.exe")

    site_packages = Path(sysconfig.get_path("purelib"))
    bootstrap = textwrap.dedent('''\
        """Load Torch DLLs before Paddle when either course imports Paddle."""
        import importlib.abc
        import os
        import sys

        os.environ.setdefault("PARL_BACKEND", "paddle")


        class TorchBeforePaddle(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "paddle":
                    sys.meta_path.remove(self)
                    import torch
                return None


        if sys.platform == "win32":
            sys.meta_path.insert(0, TorchBeforePaddle())
        ''')
    (site_packages / "_rl_framework_compat.py").write_text(bootstrap, encoding="utf-8")
    startup = site_packages / "rl_parl_backend.pth"
    startup.write_text("import _rl_framework_compat\n", encoding="ascii")
    print(f"Configured PARL backend and Torch/Paddle DLL loading order in {startup}")


if __name__ == "__main__":
    main()
