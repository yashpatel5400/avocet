"""
Compatibility shim for the robbuffet package name.

Exports everything from the underlying avocet package so existing code can
import robbuffet.* while the core package name remains avocet-cp.
"""

from avocet import *  # noqa: F401,F403
