"""
i18n loader for encrypted translation files
"""
import sys
import importlib.util
from pathlib import Path

def load_pyc_module(module_name, pyc_path):
    """Load a .pyc module using importlib"""
    spec = importlib.util.spec_from_file_location(module_name, pyc_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {pyc_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_translations():
    """Load all encrypted translation files"""
    translations = {}
    i18n_dir = Path(__file__).parent
    
    # List all .pyc files in i18n directory
    for pyc_file in i18n_dir.glob("*.pyc"):
        lang = pyc_file.stem  # Get language code from filename
        try:
            module = load_pyc_module(f"i18n_{lang}", pyc_file)
            if hasattr(module, 'data'):
                translations[lang] = module.data
        except Exception as e:
            print(f"Failed to load {pyc_file.name}: {e}")
    
    return translations

# Auto-load translations when module is imported
translations = load_translations()
