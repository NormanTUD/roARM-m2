import serial
import json
import time

class RoArmController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        """Stellt die Verbindung zum Arm her und setzt die Pegel korrekt."""
        print(f"Verbinde mit RoArm-M2-S auf {self.port}...")
        self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
        self.ser.setRTS(True)
        self.ser.setDTR(True)
        time.sleep(2)  # Boot-Zeit abwarten
        print("Verbindung erfolgreich hergestellt.")
        
        # Drehmoment standardmäßig aktivieren
        self.set_torque(True)

    def disconnect(self):
        """Schließt die serielle Verbindung sauber."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Verbindung geschlossen.")

    def _send(self, command_dict, wait_time=1.0):
        """Interne Methode zum Senden von JSON-Befehlen und Abfangen von Antworten."""
        if not self.ser or not self.ser.is_open:
            print("[Fehler]: Nicht verbunden!")
            return None

        json_cmd = json.dumps(command_dict) + '\n'
        self.ser.write(json_cmd.encode('utf-8'))
        self.ser.flush()
        
        time.sleep(wait_time)
        
        responses = []
        if self.ser.in_waiting:
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    responses.append(line)
        return responses

    def set_torque(self, enable=True):
        """Aktiviert (True) oder deaktiviert (False) die Motorenkraft."""
        cmd = {"T": 210, "cmd": 1 if enable else 0}
        return self._send(cmd, wait_time=0.5)

    def move_home(self):
        """Versucht die Standard-Ausgangsposition anzufahren."""
        return self._send({"T": 100}, wait_time=3.0)

    def get_status(self):
        """Fragt aktuelle Koordinaten, Winkel und Lastzustände ab (CMD 105)."""
        return self._send({"T": 105}, wait_time=0.5)

    def move_joints_angle(self, b=0, s=0, e=90, h=180, spd=10, acc=5):
        """
        Steuert alle Gelenke direkt über Winkel in Grad an (Befehl 122).
        b: Basis (-180 bis 180)
        s: Schulter (-90 bis 90)
        e: Ellbogen (0 bis 180, Standard 90)
        h: Greifer/Hand (45 bis 180, Standard 180)
        """
        cmd = {
            "T": 122,
            "b": b, "s": s, "e": e, "h": h,
            "spd": spd, "acc": acc
        }
        return self._send(cmd, wait_time=2.0)

    def move_cartesian(self, x, y, z, t=3.14, spd=0.25):
        """
        Bewegt die Greiferspitze zu X, Y, Z Koordinaten in mm (Befehl 104).
        t: Winkel des Greifers in Radiant (Standard 3.14)
        spd: Geschwindigkeit (z.B. 0.25)
        """
        cmd = {
            "T": 104,
            "x": x, "y": y, "z": z, "t": t,
            "spd": spd
        }
        return self._send(cmd, wait_time=3.0)

    def control_gripper(self, target_rad, spd=0, acc=0):
        """
        Steuert nur den Greifer separat in Radiant an (Befehl 106).
        Bereich ca. 1.08 (offen) bis 3.14 (geschlossen).
        """
        cmd = {
            "T": 106,
            "cmd": target_rad,
            "spd": spd,
            "acc": acc
        }
        return self._send(cmd, wait_time=1.0)
