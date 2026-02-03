import numpy as np
from collections import deque
import time

class MovementMonitor:
    def __init__(self, history_seconds=60):
        self.history_seconds = history_seconds
        # track_id -> deque of (timestamp, x, y)
        self.tracks = {}
        self.last_cleanup = time.time()

        # Umbrales (en píxeles acumulados por minuto)
        # Ajustar según la resolución de la cámara y distancia al objetivo.
        self.HIGH_ACTIVITY_THRESHOLD = 500.0
        self.LOW_ACTIVITY_THRESHOLD = 50.0

    def update(self, track_id, center_x, center_y):
        """Registra la posición actual de un pájaro."""
        now = time.time()
        if track_id not in self.tracks:
            self.tracks[track_id] = deque()

        self.tracks[track_id].append((now, center_x, center_y))

        # Limpieza periódica (cada 5 seg) para no iterar en cada frame
        if now - self.last_cleanup > 5.0:
            self._cleanup_old_tracks(now)
            self.last_cleanup = now

    def _cleanup_old_tracks(self, now):
        cutoff = now - self.history_seconds
        # Limpiar puntos viejos dentro de cada track
        for tid in list(self.tracks.keys()):
            dq = self.tracks[tid]
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            # Si el track está vacío o muy viejo (el pájaro se fue), se borra
            if not dq:
                del self.tracks[tid]
            elif (now - dq[-1][0]) > self.history_seconds:
                # Si el último punto visto es muy viejo, borrar todo el track
                del self.tracks[tid]

    def get_activity_level(self, track_id):
        """Calcula nivel de actividad para un ID específico."""
        if track_id not in self.tracks:
            return "Desconocido", 0.0

        dq = self.tracks[track_id]
        if len(dq) < 2:
            return "Sedentario", 0.0

        total_dist = 0.0
        # Calcular distancia euclidiana acumulada
        for i in range(1, len(dq)):
            _, x1, y1 = dq[i-1]
            _, x2, y2 = dq[i]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_dist += dist

        if total_dist > self.HIGH_ACTIVITY_THRESHOLD:
            return "Muy Activo", total_dist
        elif total_dist > self.LOW_ACTIVITY_THRESHOLD:
            return "Normal", total_dist
        else:
            return "Sedentario", total_dist

    def get_global_activity(self):
        """Retorna el estado general basado en el pájaro más activo."""
        levels = []
        dists = []

        # Si no hay tracks recientes
        if not self.tracks:
            return "Sin datos", 0.0

        for tid in self.tracks:
            lvl, dist = self.get_activity_level(tid)
            levels.append(lvl)
            dists.append(dist)

        if not levels:
            return "Sin datos", 0.0

        # Prioridad: Muy Activo > Normal > Sedentario
        if "Muy Activo" in levels:
            return "Alta Actividad", max(dists)
        elif "Normal" in levels:
            return "Actividad Normal", max(dists)
        else:
            return "Tranquilo", max(dists)
