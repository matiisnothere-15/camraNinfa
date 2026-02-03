import time

class HealthMonitor:
    def __init__(self):
        # Timestamps de última vez visto
        # Inicializamos en 'ahora' para evitar alertas falsas al arrancar el sistema
        self.last_eating = time.time()
        self.last_drinking = time.time()

        # Configuración de umbrales (en segundos)
        # Ejemplo: 4 horas = 4 * 3600 = 14400
        self.THRESHOLD_EATING = 14400
        self.THRESHOLD_DRINKING = 14400

    def register_action(self, action):
        """Actualiza los timestamps según la acción detectada."""
        if not action:
            return

        now = time.time()
        if "Alimentacion" in action:
            self.last_eating = now
        elif "Hidratacion" in action:
            self.last_drinking = now

    def check_health(self):
        """Retorna una lista de alertas si se exceden los tiempos."""
        alerts = []
        now = time.time()

        elapsed_eating = now - self.last_eating
        elapsed_drinking = now - self.last_drinking

        if elapsed_eating > self.THRESHOLD_EATING:
            hours = elapsed_eating / 3600
            alerts.append(f"⚠️ SIN COMER: {hours:.1f}h")

        if elapsed_drinking > self.THRESHOLD_DRINKING:
            hours = elapsed_drinking / 3600
            alerts.append(f"⚠️ SIN BEBER: {hours:.1f}h")

        return alerts

    def get_stats(self):
        now = time.time()
        return {
            "eating_sec": now - self.last_eating,
            "drinking_sec": now - self.last_drinking
        }
