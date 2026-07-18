#!/usr/bin/env python3
"""
apply_patch.py - Wendet SEARCH/REPLACE Patches auf eine Datei an.

Usage:
    python apply_patch.py <target_file> [patch_file]
    
    Wenn kein patch_file angegeben wird, liest es von stdin.
    Bei stdin-Eingabe: 2 Leerzeilen hintereinander beenden die Eingabe.

Patch-Format:
    <<<<<<< SEARCH
    alter code hier
    =======
    neuer code hier
    >>>>>>> REPLACE
"""

import sys
import re


def read_from_stdin():
    """Liest von stdin bis 2 aufeinanderfolgende Leerzeilen kommen."""
    print("Patch eingeben (2 Leerzeilen zum Beenden):", file=sys.stderr)
    lines = []
    empty_count = 0

    for line in sys.stdin:
        if line.strip() == '':
            empty_count += 1
            if empty_count >= 2:
                # Die letzte Leerzeile wieder entfernen die zum Zählen diente
                if lines and lines[-1].strip() == '':
                    lines.pop()
                break
        else:
            empty_count = 0
        lines.append(line)

    return ''.join(lines)


def read_patch(patch_file=None):
    """Liest Patch aus Datei oder stdin."""
    if patch_file:
        with open(patch_file, 'r') as f:
            return f.read()
    else:
        return read_from_stdin()


def apply_patches(content, patch_text):
    """Wendet alle SEARCH/REPLACE Blöcke auf den Content an."""
    blocks = re.findall(
        r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
        patch_text, re.DOTALL
    )

    if not blocks:
        print("FEHLER: Keine SEARCH/REPLACE Blöcke gefunden.", file=sys.stderr)
        return content, 0, 0

    applied = 0
    failed = 0

    for search, replace in blocks:
        if search in content:
            content = content.replace(search, replace, 1)
            applied += 1
            print(f"  ✓ Block angewendet ({search[:50].strip()}...)", file=sys.stderr)
        else:
            failed += 1
            print(f"  ✗ Block NICHT gefunden:", file=sys.stderr)
            print(f"    {search[:80].strip()}...", file=sys.stderr)

    return content, applied, failed


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_file> [patch_file]", file=sys.stderr)
        sys.exit(1)

    target_file = sys.argv[1]
    patch_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Zieldatei lesen
    try:
        with open(target_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"FEHLER: Datei '{target_file}' nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    # Patch lesen
    patch_text = read_patch(patch_file)

    if not patch_text.strip():
        print("FEHLER: Kein Patch-Input erhalten.", file=sys.stderr)
        sys.exit(1)

    # Patches anwenden
    new_content, applied, failed = apply_patches(content, patch_text)

    # Ergebnis schreiben
    with open(target_file, 'w') as f:
        f.write(new_content)

    print(f"\nFertig: {applied} angewendet, {failed} fehlgeschlagen.", file=sys.stderr)

    if failed > 0:
        sys.exit(2)


if __name__ == '__main__':
    main()
