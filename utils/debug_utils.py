"""
Debug utilities for DaxNodes
Only shows debug output when DAXNODES_DEBUG environment variable is set
"""

import os

def is_debug_enabled():
    """Check if debug mode is enabled via environment variable"""
    return os.environ.get('DAXNODES_DEBUG', '').lower() in ('true', '1', 'yes', 'on')

def debug_print(*args, **kwargs):
    """Print debug message only if debug mode is enabled"""
    if is_debug_enabled():
        print("[DEBUG]", *args, **kwargs)

# Global debug flag for easy access
DEBUG_ENABLED = is_debug_enabled()