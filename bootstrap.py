#!/usr/bin/env python3
"""bootstrap.py - UV Environment Bootstrapping

Stellt sicher dass das Script in einer UV-verwalteten Umgebung läuft.
Wird von allen ausführbaren Scripts importiert.

Usage in anderen Dateien:
    from bootstrap import ensure_uv
    ensure_uv()
"""

import os
import sys


def ensure_uv():
    """Startet das Script in einer UV-Umgebung neu falls nötig.
    
    Wenn bereits in UV-Umgebung: Tut nichts.
    Wenn nicht: Ruft `uv run` auf und ersetzt den aktuellen Prozess.
    
    Muss VOR allen anderen Imports aufgerufen werden!
    """
    if os.environ.get("_UV_SAFE_ENV") == "1":
        return
    
    os.environ["_UV_SAFE_ENV"] = "1"
    
    from datetime import datetime, timedelta, timezone
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
    
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
    except FileNotFoundError:
        print("uv nicht installiert. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
