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

def ensure_dialout():
    """Warnt wenn der User nicht in der dialout-Gruppe ist."""
    import grp, os
    try:
        dialout_gid = grp.getgrnam("dialout").gr_gid
        if dialout_gid not in os.getgroups():
            print("⚠️  WARNUNG: Nicht in der 'dialout'-Gruppe!")
            print("   Serieller Zugriff wird fehlschlagen.")
            print("   Fix: sudo usermod -aG dialout $USER")
            print("   Danach: Neu einloggen oder 'newgrp dialout'")
    except KeyError:
        pass  # Gruppe existiert nicht (z.B. macOS)

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
    else:
        ensure_dialout()
    
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
    except FileNotFoundError:
        print("uv nicht installiert. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
