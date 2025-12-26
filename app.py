"""
Minimal app loader for Omni-Image-Editor
This app loads the compiled, obfuscated modules from __lib__
"""
import sys
from pathlib import Path
import importlib.util

# Add __lib__ to path to import compiled modules
lib_dir = Path(__file__).parent / "__lib__"
if not lib_dir.exists():
    raise RuntimeError(f"Compiled library directory not found: {lib_dir}")

sys.path.insert(0, str(lib_dir))

def load_pyc_module(module_name, pyc_path):
    """Load a .pyc module using importlib"""
    spec = importlib.util.spec_from_file_location(module_name, pyc_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {pyc_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    # Load compiled modules
    util_module = load_pyc_module("util", lib_dir / "util.pyc")
    app_module = load_pyc_module("app", lib_dir / "app.pyc")

    # Create and launch app
    app = app_module.create_app()
    app.queue(
        default_concurrency_limit=20,
        max_size=50,
        api_open=False
    )
    app.launch(
        server_name="0.0.0.0",
        show_error=True,
        quiet=False,
        max_threads=40,
        height=800,
        favicon_path=None
    )

except ImportError as e:
    print(f"Failed to import compiled modules: {e}")
    print("Make sure to run build_encrypted.py first to compile the modules")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"Error running app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
