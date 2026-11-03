from __future__ import annotations
import os

# # Acces to DLLs used by MinGW used by Clion
# compiler_dll_dir = "C:/JetBrains/CLion/bin/mingw/bin"
# if os.path.exists(compiler_dll_dir):
#     os.add_dll_directory(compiler_dll_dir)
#
# # Double check .venv path is known
# venv_bin_dir = "D:/01b-Dev/Python/Projects/Discrete_models/.venv/Scripts"
# if os.path.exists(venv_bin_dir):
#     os.add_dll_directory(venv_bin_dir)

try:
    import discrete_models
    print("Acces granted.")
    print("Module contains :", dir(discrete_models))
except ImportError as e:
    print("Acces denied. :", e)

import discrete_models

__all__ = ["discrete_models"]