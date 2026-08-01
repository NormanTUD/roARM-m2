def parse_roarm_file(filepath: str) -> dict:
    """Parses a .roarm file including LED events and waypoint-mode detection."""
    waypoints = []
    events = []
    config = {"hz": 20, "threshold": 0.3, "gravity_comp": 1, "mode": "continuous"}
    start_pos = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#OFFSET"):
                continue
            if line.startswith("#CONFIG"):
                _parse_config_line(line, config)
            elif line.startswith("#START_POS"):
                start_pos = _parse_start_pos_line(line)
            elif line.startswith("#"):
                continue
            elif line.startswith("MOVE"):
                waypoints.append(_parse_move_line(line))
            elif line.startswith("GRIPPER"):
                events.append(_parse_gripper_line(line))
            elif line.startswith("LED"):
                events.append(_parse_led_line(line))

    if start_pos is None:
        start_pos = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}

    config["mode"] = str(config.get("mode", "continuous")).strip().lower()

    return {
        "waypoints": waypoints,
        "events": events,
        "config": config,
        "start_pos": start_pos,
    }


def is_waypoint_recording(parsed: dict) -> bool:
    """True if the recording was captured in waypoint mode."""
    return parsed.get("config", {}).get("mode", "continuous") == "waypoint"


def waypoint_poses(parsed: dict) -> list:
    """Return waypoints as a list of pure joint poses (no timing data).

    For continuous recordings, the playback timing already comes from
    `parsed['waypoints']`; this helper strips the timestamps and returns
    just {b,s,e,h} dicts, suitable for TrapezoidTrajectory.
    """
    return [
        {"b": wp["b"], "s": wp["s"], "e": wp["e"], "h": wp["h"]}
        for wp in parsed.get("waypoints", [])
    ]


def _parse_config_line(line: str, config: dict):
    parts = line.split(" ", 1)
    if len(parts) == 2:
        key, val = parts[1].split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            config[key] = float(val)
        except ValueError:
            config[key] = val


def _parse_start_pos_line(line: str) -> dict:
    vals = {}
    for p in line.split()[1:]:
        k, v = p.split("=")
        vals[k] = float(v)
    return vals


def _parse_move_line(line: str) -> dict:
    vals = {}
    for p in line.split()[1:]:
        k, v = p.split("=")
        vals[k] = float(v)
    return {
        "t": vals.get("t", 0.0), "b": vals.get("b", 0.0),
        "s": vals.get("s", 0.0), "e": vals.get("e", 90.0),
        "h": vals.get("h", 180.0),
    }


def _parse_gripper_line(line: str) -> dict:
    parts = line.split()
    cmd = parts[1] if len(parts) > 1 else "OPEN"
    t = _extract_time_from_parts(parts)
    return {"t": t, "cmd": cmd}


def _parse_led_line(line: str) -> dict:
    parts = line.split()
    cmd = "LED_ON" if (len(parts) > 1 and parts[1] == "ON") else "LED_OFF"
    t = _extract_time_from_parts(parts)
    return {"t": t, "cmd": cmd}


def _extract_time_from_parts(parts: list) -> float:
    for p in parts:
        if p.startswith("t="):
            return float(p.split("=")[1])
    return 0.0
