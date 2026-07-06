from roarm_controller import RoArmController
import time

# Instanz erstellen
arm = RoArmController(port='/dev/ttyUSB0', baudrate=115200)

try:
    # 1. Verbindung aufbauen
    arm.connect()
    
    # 2. Status abfragen (zeigt Winkel und aktuelle Millimeter-Koordinaten)
    print("\n--- Aktueller Status ---")
    status = arm.get_status()
    print(status)
    
    # 3. Gelenke bewegen (Basis 20 Grad, Ellbogen auf 100 Grad)
    print("\n--- Gelenke bewegen ---")
    arm.move_joints_angle(b=20, s=0, e=100, h=180, spd=15)
    
    # 4. Greifer testen (Öffnen)
    print("\n--- Greifer öffnen ---")
    arm.control_gripper(target_rad=1.5) 
    time.sleep(1)
    
    # 5. Greifer testen (Schließen)
    print("\n--- Greifer schließen ---")
    arm.control_gripper(target_rad=3.14)

    # 6. Zurück in eine sichere Standard-Winkelposition
    print("\n--- Zurück zur Parkposition ---")
    arm.move_joints_angle(b=0, s=0, e=90, h=180, spd=15)

finally:
    # Sauberes Beenden
    arm.disconnect()
