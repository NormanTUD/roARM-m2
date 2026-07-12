#!/usr/bin/env python3
"""ui.py - Rich-basierte UI-Schicht für RoArm-M2-S

Ersetzt alle print()-Aufrufe durch hübsche Rich-Ausgaben:
- Farbige Status-Panels
- Live-Tabellen für Gelenkpositionen
- Progress-Bars für Streaming/Kalibrierung
- Strukturierte Fehler-Anzeige
- Live-Dashboard während Playback
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    SpinnerColumn, TaskProgressColumn
)
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.style import Style
from rich.theme import Theme
from rich import box

# ============================================================
# THEME & CONSOLE
# ============================================================

ROARM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "joint.b": "bright_blue",
    "joint.s": "bright_magenta",
    "joint.e": "bright_yellow",
    "joint.h": "bright_cyan",
    "safety": "bold red on dark_red",
    "thermal.ok": "green",
    "thermal.warm": "yellow",
    "thermal.hot": "bold yellow",
    "thermal.critical": "bold white on red",
    "speed.slow": "blue",
    "speed.normal": "green",
    "speed.fast": "red",
    "header": "bold white on blue",
})

console = Console(theme=ROARM_THEME)


# ============================================================
# HEADER & BANNER
# ============================================================

def print_banner(mode: str, subtitle: str = ""):
    """Zeigt ein hübsches Banner für den jeweiligen Modus."""
    mode_icons = {
        "play": "▶️",
        "teach": "🎓",
        "calibrate": "🎯",
        "safety": "🛡️",
    }
    icon = mode_icons.get(mode, "🤖")
    
    title = f"{icon} RoArm-M2-S — {mode.upper()} MODE"
    
    content = Text()
    content.append(title, style="bold white")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    
    console.print(Panel(
        content,
        border_style="blue",
        box=box.DOUBLE,
        padding=(1, 2),
    ))


def print_section(title: str, icon: str = "─"):
    """Abschnitts-Trenner."""
    console.print(Rule(f" {title} ", style="dim blue"))


# ============================================================
# GELENK-TABELLEN
# ============================================================

def joint_table(positions: dict, title: str = "Gelenkpositionen",
                target: dict = None, show_error: bool = True) -> Table:
    """
    Erstellt eine hübsche Tabelle mit Gelenkpositionen.
    Optional mit Soll-Werten und Fehler-Spalte.
    """
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    
    table.add_column("Gelenk", style="bold", width=10)
    table.add_column("Ist [°]", justify="right", width=10)
    
    if target:
        table.add_column("Soll [°]", justify="right", width=10)
    if target and show_error:
        table.add_column("Fehler [°]", justify="right", width=12)
        table.add_column("Status", justify="center", width=8)
    
    joint_names = {
        "b": ("Base", "joint.b"),
        "s": ("Shoulder", "joint.s"),
        "e": ("Elbow", "joint.e"),
        "h": ("Hand", "joint.h"),
    }
    
    for j, (name, style) in joint_names.items():
        if j not in positions:
            continue
            
        row = [f"[{style}]{name}[/]", f"{positions[j]:7.2f}"]
        
        if target:
            row.append(f"{target[j]:7.2f}")
            
        if target and show_error:
            err = positions[j] - target[j]
            abs_err = abs(err)
            
            if abs_err < 0.3:
                err_style = "green"
                status = "✅"
            elif abs_err < 1.0:
                err_style = "yellow"
                status = "⚠️"
            else:
                err_style = "red"
                status = "❌"
            
            row.append(f"[{err_style}]{err:+.3f}[/]")
            row.append(status)
        
        table.add_row(*row)
    
    return table


def print_position(positions: dict, title: str = "Position", **kwargs):
    """Shortcut: Druckt eine Positions-Tabelle."""
    console.print(joint_table(positions, title=title, **kwargs))


def print_position_inline(positions: dict, prefix: str = ""):
    """Kompakte einzeilige Positions-Anzeige."""
    text = Text(prefix)
    for j, style in [("b", "joint.b"), ("s", "joint.s"), 
                     ("e", "joint.e"), ("h", "joint.h")]:
        if j in positions:
            text.append(f" {j}=", style="dim")
            text.append(f"{positions[j]:7.2f}°", style=style)
    console.print(text)


# ============================================================
# SAFETY STATUS
# ============================================================

def safety_status_table(stats: dict, thermal_temp: float = None,
                        thermal_status: str = "OK") -> Table:
    """Zeigt den aktuellen Safety-Status als Tabelle."""
    table = Table(
        title="🛡️ Safety Status",
        box=box.SIMPLE_HEAVY,
        show_header=False,
        border_style="dim red" if stats.get("emergency_stop") else "dim green",
    )
    
    table.add_column("Parameter", style="bold", width=25)
    table.add_column("Wert", justify="right", width=20)
    table.add_column("Status", justify="center", width=8)
    
    # Emergency Stop
    e_stop = stats.get("emergency_stop", False)
    table.add_row(
        "Emergency Stop",
        "[bold red]AKTIV[/]" if e_stop else "[green]Inaktiv[/]",
        "🚨" if e_stop else "✅"
    )
    
    # Befehle gesendet
    table.add_row(
        "Befehle gesendet",
        str(stats.get("total_commands", 0)),
        ""
    )
    
    # Kontinuierliche Bewegung
    cont_s = stats.get("continuous_move_s", 0)
    if cont_s > 60:
        cont_style = "bold red"
        cont_status = "⚠️"
    elif cont_s > 30:
        cont_style = "yellow"
        cont_status = "🔶"
    else:
        cont_style = "green"
        cont_status = "✅"
    table.add_row(
        "Kontinuierliche Bewegung",
        f"[{cont_style}]{cont_s:.1f}s[/]",
        cont_status
    )
    
    # Temperatur
    if thermal_temp is not None:
        temp_styles = {
            "OK": "thermal.ok",
            "WARM": "thermal.warm",
            "HOT": "thermal.hot",
            "CRITICAL": "thermal.critical",
        }
        temp_icons = {"OK": "✅", "WARM": "🌡️", "HOT": "⚠️", "CRITICAL": "🚨"}
        style = temp_styles.get(thermal_status, "white")
        table.add_row(
            "Temperatur (geschätzt)",
            f"[{style}]~{thermal_temp:.0f}°C ({thermal_status})[/]",
            temp_icons.get(thermal_status, "")
        )
    
    return table


def print_safety_violation(reason: str):
    """Zeigt eine Safety-Violation prominent an."""
    console.print(Panel(
        f"[bold red]⛔ SAFETY VIOLATION[/]\n\n{reason}\n\n[dim]Befehl wurde NICHT gesendet![/]",
        border_style="red",
        box=box.HEAVY,
        padding=(1, 2),
    ))


def print_emergency_stop(reason: str):
    """Zeigt einen Emergency Stop an."""
    console.print()
    console.print(Panel(
        f"[bold white on red] 🚨 EMERGENCY STOP 🚨 [/]\n\n"
        f"[bold]{reason}[/]\n\n"
        f"[dim]Arm wird gestoppt und Torque ausgeschaltet.[/]",
        border_style="bold red",
        box=box.DOUBLE,
        padding=(1, 3),
    ))
    console.print()


# ============================================================
# STREAMING PLAYBACK LIVE-DISPLAY
# ============================================================

class PlaybackDisplay:
    """
    Live-Dashboard während des Streamings.
    Zeigt in Echtzeit:
    - Progress-Bar
    - Aktuelle Position
    - Speed-Profil
    - Safety-Status
    """
    
    def __init__(self, total_duration: float, stream_hz: int):
        self._duration = total_duration
        self._hz = stream_hz
        self._live = None
        self._progress = None
        self._task_id = None
        
    def _build_layout(self, elapsed: float, target: dict, 
                      speed_factor: float, commands_sent: int,
                      skipped: int, thermal_temp: float = None,
                      thermal_status: str = "OK",
                      tracking_error: float = None) -> Table:
        """Baut das Live-Display zusammen."""
        
        # Haupt-Layout als verschachtelte Tabelle
        main = Table(box=box.SIMPLE, show_header=False, show_edge=False, 
                     padding=(0, 1))
        main.add_column(ratio=1)
        
        # === Progress Bar ===
        pct = min(100.0, (elapsed / self._duration) * 100)
        bar_width = 40
        filled = int(pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        progress_text = Text()
        progress_text.append("  ▶ ", style="bold green")
        progress_text.append(f"[{bar}]", style="bright_blue")
        progress_text.append(f" {pct:5.1f}%", style="bold")
        progress_text.append(f"  {elapsed:6.2f}s / {self._duration:.2f}s", style="dim")
        main.add_row(progress_text)
        
        # === Position ===
        pos_text = Text()
        pos_text.append("  📍 ", style="dim")
        for j, style in [("b", "joint.b"), ("s", "joint.s"), 
                         ("e", "joint.e"), ("h", "joint.h")]:
            if j in target:
                pos_text.append(f"{j}=", style="dim")
                pos_text.append(f"{target[j]:7.2f}° ", style=style)
        main.add_row(pos_text)
        
        # === Speed & Stats ===
        stats_text = Text()
        stats_text.append("  ⚡ ", style="dim")
        
        # Speed-Faktor mit Farbe
        if speed_factor > 1.5:
            spd_style = "speed.fast"
        elif speed_factor > 0.8:
            spd_style = "speed.normal"
        else:
            spd_style = "speed.slow"
        stats_text.append(f"v={speed_factor:.2f}x", style=spd_style)
        
        stats_text.append(f"  │  Cmds: {commands_sent}", style="dim")
        stats_text.append(f"  │  Skip: {skipped}", style="dim")
        
        if thermal_temp is not None:
            temp_style = {
                "OK": "thermal.ok", "WARM": "thermal.warm",
                "HOT": "thermal.hot", "CRITICAL": "thermal.critical"
            }.get(thermal_status, "white")
            stats_text.append(f"  │  ", style="dim")
            stats_text.append(f"🌡️ {thermal_temp:.0f}°C", style=temp_style)
        
        if tracking_error is not None:
            err_style = "green" if tracking_error < 3.0 else "yellow" if tracking_error < 6.0 else "red"
            stats_text.append(f"  │  ", style="dim")
            stats_text.append(f"Δ={tracking_error:.1f}°", style=err_style)
        
        main.add_row(stats_text)
        
        return main
    
    def start(self):
        """Startet das Live-Display."""
        self._live = Live(
            console=console,
            refresh_per_second=15,
            transient=True,
        )
        self._live.start()
    
    def update(self, elapsed: float, target: dict, speed_factor: float,
               commands_sent: int, skipped: int, **kwargs):
        """Aktualisiert das Live-Display."""
        if self._live:
            layout = self._build_layout(
                elapsed, target, speed_factor, commands_sent, skipped, **kwargs
            )
            self._live.update(layout)
    
    def stop(self):
        """Stoppt das Live-Display."""
        if self._live:
            self._live.stop()


# ============================================================
# KALIBRIERUNGS-FORTSCHRITT
# ============================================================

def calibration_progress(total_measurements: int) -> Progress:
    """Erstellt eine Progress-Bar für die Kalibrierung."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )


def calibration_pose_table(pose_index: int, total_poses: int,
                           repeat: int, total_repeats: int,
                           commanded: dict, measured: dict = None,
                           error: dict = None) -> Table:
    """Zeigt den Status einer Kalibrierungs-Pose."""
    table = Table(
        title=f"Pose {pose_index+1}/{total_poses} • Wiederholung {repeat+1}/{total_repeats}",
        box=box.ROUNDED,
        border_style="blue",
    )
    
    table.add_column("Gelenk", style="bold", width=10)
    table.add_column("Soll [°]", justify="right", width=10)
    
    if measured:
        table.add_column("Ist [°]", justify="right", width=10)
    if error:
        table.add_column("Fehler [°]", justify="right", width=12)
    
    for j in ["b", "s", "e"]:
        row = [f"[joint.{j}]{j.upper()}[/]", f"{commanded[j]:.1f}"]
        if measured:
            row.append(f"{measured[j]:.3f}")
        if error:
            err_val = error[j]
            style = "green" if abs(err_val) < 0.3 else "yellow" if abs(err_val) < 1.0 else "red"
            row.append(f"[{style}]{err_val:+.3f}[/]")
        table.add_row(*row)
    
    return table


# ============================================================
# TEACH MODE DISPLAY
# ============================================================

class TeachDisplay:
    """Live-Anzeige während der Aufnahme."""
    
    def __init__(self, hz: int, threshold: float, gravity_comp: bool):
        self._hz = hz
        self._threshold = threshold
        self._gravity_comp = gravity_comp
        self._live = None
    
    def _build_display(self, elapsed: float, waypoint_count: int,
                       current_pos: dict, last_recorded: dict = None) -> Table:
        """Baut die Teach-Anzeige."""
        main = Table(box=box.SIMPLE, show_header=False, show_edge=False)
        main.add_column(ratio=1)
        
        # Header
        header = Text()
        header.append("  🔴 REC ", style="bold red")
        header.append(f"  {elapsed:6.2f}s", style="bold")
        header.append(f"  │  WP: {waypoint_count}", style="cyan")
        header.append(f"  │  {self._hz}Hz", style="dim")
        if self._gravity_comp:
            header.append("  │  GravComp ✓", style="dim green")
        main.add_row(header)
        
        # Position
        if current_pos:
            pos_text = Text()
            pos_text.append("  📍 ", style="dim")
            for j, style in [("b", "joint.b"), ("s", "joint.s"),
                             ("e", "joint.e"), ("h", "joint.h")]:
                pos_text.append(f"{j}=", style="dim")
                pos_text.append(f"{current_pos[j]:7.2f}° ", style=style)
            main.add_row(pos_text)
        
        # Delta zum letzten aufgezeichneten Punkt
        if current_pos and last_recorded:
            delta = max(abs(current_pos[j] - last_recorded[j]) for j in ["b", "s", "e", "h"])
            delta_style = "green" if delta > self._threshold else "dim"
            delta_text = Text()
            delta_text.append(f"  Δ={delta:.2f}°", style=delta_style)
            if delta > self._threshold:
                delta_text.append(" (recording)", style="green")
            else:
                delta_text.append(" (still)", style="dim")
            main.add_row(delta_text)
        
        # Controls
        controls = Text()
        controls.append("\n  [ENTER]=Stop  [g]=Gripper  ", style="dim")
        main.add_row(controls)
        
        return main
    
    def start(self):
        self._live = Live(console=console, refresh_per_second=10, transient=True)
        self._live.start()
    
    def update(self, elapsed: float, waypoint_count: int,
               current_pos: dict, last_recorded: dict = None):
        if self._live:
            display = self._build_display(elapsed, waypoint_count, current_pos, last_recorded)
            self._live.update(display)
    
    def stop(self):
        if self._live:
            self._live.stop()


# ============================================================
# ERGEBNIS-ZUSAMMENFASSUNGEN
# ============================================================

def playback_summary(duration_actual: float, duration_planned: float,
                     commands_sent: int, skipped: int,
                     final_error: float = None,
                     thermal_temp: float = None,
                     rate_limiter_violations: int = 0):
    """Zeigt eine hübsche Zusammenfassung nach dem Playback."""
    
    table = Table(
        title="📊 Playback Zusammenfassung",
        box=box.ROUNDED,
        border_style="green",
    )
    
    table.add_column("Metrik", style="bold", width=25)
    table.add_column("Wert", justify="right", width=20)
    
    table.add_row("Dauer (Ist)", f"{duration_actual:.2f}s")
    table.add_row("Dauer (Soll)", f"{duration_planned:.2f}s")
    table.add_row("Befehle gesendet", str(commands_sent))
    table.add_row("Übersprungen (< Δ)", str(skipped))
    table.add_row("Effektive Rate", f"{commands_sent/max(duration_actual,0.1):.1f} Hz")
    
    if final_error is not None:
        err_style = "green" if final_error < 0.5 else "yellow" if final_error < 1.5 else "red"
        table.add_row("Endposition Fehler", f"[{err_style}]{final_error:.3f}°[/]")
    
    if thermal_temp is not None:
        table.add_row("Temperatur (Ende)", f"~{thermal_temp:.0f}°C")
    
    if rate_limiter_violations > 0:
        table.add_row("Rate-Limiter Eingriffe", f"[yellow]{rate_limiter_violations}[/]")
    
    console.print()
    console.print(table)


def calibration_summary(residuals: dict, total_time: float,
                        n_poses: int, n_repeats: int,
                        error_stats: dict = None):
    """Zeigt Kalibrierungs-Ergebnis."""
    
    # Hauptergebnis
    import numpy as np
    total_rms = np.sqrt(np.mean([r**2 for r in residuals.values()]))
    
    if total_rms < 0.5:
        quality = "[bold green]SEHR GUT ✅[/]"
        border = "green"
    elif total_rms < 1.0:
        quality = "[yellow]AKZEPTABEL ⚠️[/]"
        border = "yellow"
    else:
        quality = "[bold red]SCHLECHT ❌[/]"
        border = "red"
    
    table = Table(
        title="🎯 Kalibrierungs-Ergebnis",
        box=box.DOUBLE,
        border_style=border,
    )
    
    table.add_column("Gelenk", style="bold", width=10)
    table.add_column("RMS Residuum [°]", justify="right", width=18)
    table.add_column("Qualität", justify="center", width=12)
    
    for joint, rms in residuals.items():
        q = "✅" if rms < 0.3 else "⚠️" if rms < 0.7 else "❌"
        style = "green" if rms < 0.3 else "yellow" if rms < 0.7 else "red"
        table.add_row(f"[joint.{joint}]{joint.upper()}[/]", 
                      f"[{style}]{rms:.4f}[/]", q)
    
    table.add_row("─" * 8, "─" * 16, "─" * 10)
    table.add_row("[bold]Gesamt[/]", f"[bold]{total_rms:.4f}[/]", quality)
    
    console.print()
    console.print(table)
    
    # Meta-Info
    meta = Table(box=box.SIMPLE, show_header=False, show_edge=False)
    meta.add_column(width=30)
    meta.add_column(justify="right", width=20)
    meta.add_row("Gesamtzeit", f"{total_time:.1f}s")
    meta.add_row("Posen × Wiederholungen", f"{n_poses} × {n_repeats}")
    meta.add_row("Messungen gesamt", f"{n_poses * n_repeats}")
    console.print(meta)


# ============================================================
# UTILITY PRINTS
# ============================================================

def print_success(msg: str):
    console.print(f"  [success]✅ {msg}[/]")

def print_warning(msg: str):
    console.print(f"  [warning]⚠️  {msg}[/]")

def print_error(msg: str):
    console.print(f"  [error]❌ {msg}[/]")

def print_info(msg: str):
    console.print(f"  [info]ℹ️  {msg}[/]")

def print_step(step: int, total: int, msg: str):
    console.print(f"  [dim][{step}/{total}][/] {msg}")


# ============================================================
# CONNECTION STATUS
# ============================================================

def print_connection_status(port: str, success: bool, 
                            safety_features: list = None):
    """Zeigt Verbindungsstatus mit Safety-Features."""
    if success:
        content = Text()
        content.append(f"✅ Verbunden mit ", style="green")
        content.append(f"{port}", style="bold cyan")
        
        if safety_features:
            content.append("\n\n🛡️ Safety-Layer aktiv:", style="dim")
            for feat in safety_features:
                content.append(f"\n   • {feat}", style="dim")
        
        console.print(Panel(content, border_style="green", box=box.ROUNDED))
    else:
        console.print(Panel(
            f"[bold red]❌ Verbindung fehlgeschlagen: {port}[/]",
            border_style="red",
            box=box.ROUNDED,
        ))


# ============================================================
# TRAJECTORY INFO
# ============================================================

def print_trajectory_info(n_waypoints: int, duration_original: float,
                          duration_smooth: float, stream_hz: int,
                          speed_stats: dict, joint_ranges: dict,
                          has_offset: bool = False, offset: dict = None):
    """Zeigt Trajektorien-Info nach dem Laden."""
    
    # Haupt-Info
    info_table = Table(
        title="📋 Trajektorie geladen",
        box=box.ROUNDED,
        border_style="cyan",
    )
    info_table.add_column("Parameter", style="bold", width=25)
    info_table.add_column("Wert", justify="right", width=25)
    
    info_table.add_row("Wegpunkte", str(n_waypoints))
    info_table.add_row("Original-Dauer", f"{duration_original:.2f}s")
    info_table.add_row("Geglättete Dauer", f"{duration_smooth:.2f}s")
    info_table.add_row("Stream-Rate", f"{stream_hz} Hz")
    info_table.add_row("Erwartete Befehle", f"~{int(duration_smooth * stream_hz)}")
    info_table.add_row("Speed min/max/avg", 
                       f"{speed_stats['min']:.2f}x / {speed_stats['max']:.2f}x / {speed_stats['avg']:.2f}x")
    
    console.print(info_table)
    
    # Gelenkbereiche
    range_table = Table(
        title="Gelenkbereiche",
        box=box.SIMPLE,
        show_header=True,
    )
    range_table.add_column("Gelenk", style="bold", width=10)
    range_table.add_column("Min [°]", justify="right", width=10)
    range_table.add_column("Max [°]", justify="right", width=10)
    range_table.add_column("Δ [°]", justify="right", width=10)
    
    for j in ["b", "s", "e", "h"]:
        if j in joint_ranges:
            r = joint_ranges[j]
            range_table.add_row(
                f"[joint.{j}]{j.upper()}[/]",
                f"{r['min']:.2f}",
                f"{r['max']:.2f}",
                f"{r['max'] - r['min']:.2f}"
            )
    
    console.print(range_table)
    
    # Offset
    if has_offset and offset:
        offset_text = Text("  📐 Offset-Korrektur: ")
        for j in ["b", "s", "e", "h"]:
            offset_text.append(f"Δ{j}={offset[j]:+.3f}° ", 
                              style="yellow" if abs(offset[j]) > 0.5 else "dim")
        console.print(offset_text)


def print_preflight_check(is_safe: bool, violations: list = None):
    """Zeigt Pre-Flight Check Ergebnis."""
    if is_safe:
        console.print(Panel(
            "[bold green]✈️  Pre-Flight Check: BESTANDEN ✅[/]\n"
            "[dim]Alle Punkte innerhalb der sicheren Grenzen.[/]",
            border_style="green",
            box=box.ROUNDED,
        ))
    else:
        content = Text()
        content.append("✈️  Pre-Flight Check: FEHLGESCHLAGEN ❌\n\n", style="bold red")
        if violations:
            for v in violations[:10]:
                content.append(f"  ⚠️  {v}\n", style="yellow")
            if len(violations) > 10:
                content.append(f"\n  ... und {len(violations)-10} weitere", style="dim")
        content.append("\n\n[dim]Trajektorie wird NICHT abgespielt.[/]")
        
        console.print(Panel(content, border_style="red", box=box.HEAVY))
