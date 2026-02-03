import cv2
import time
import os
import threading
from pathlib import Path
import importlib.util
import importlib

import numpy as np

# Intentos de importación robustos
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

from ultralytics import YOLO  # type: ignore

# --- NUEVOS MÓDULOS ---
from modulos.audio_analysis import AudioMonitor
from modulos.movement_analysis import MovementMonitor
from modulos.health_monitor import HealthMonitor

# 1. CARGA DE ENTORNO
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Intentamos cargar el módulo de BD
try:
    from modulos.base_datos import insertar_registro

    DB_ACTIVA = True
    print("✅ Base de Datos: CONECTADA")
except Exception as e:
    print(f"⚠️ Base de Datos: DESCONECTADA ({e})")
    insertar_registro = None
    DB_ACTIVA = False


def guardar_async(*args):
    """Ejecuta la inserción en BD en un hilo separado para no bloquear el video."""
    if (not DB_ACTIVA) or (insertar_registro is None):
        return

    func = insertar_registro

    def tarea():
        try:
            func(*args)
        except Exception as e:
            print(f"   └── ❌ Error guardando en hilo: {e}")

    t = threading.Thread(target=tarea, daemon=True)
    t.start()


def _obtener_url_rtsp_tapo() -> str | None:
    tapo_usuario = os.getenv("TAPO_USER")
    tapo_pass = os.getenv("TAPO_PASS")
    tapo_ip = os.getenv("TAPO_IP")

    if not tapo_usuario or not tapo_pass or not tapo_ip:
        print("❌ Faltan credenciales RTSP en .env")
        return None
    return f"rtsp://{tapo_usuario}:{tapo_pass}@{tapo_ip}:554/stream2"


def dibujar_hud(frame, accion, fps, audio_stats, move_status, mood, health_alerts):
    """Dibuja una interfaz moderna sobre el video."""
    alto, ancho = frame.shape[:2]

    # FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (ancho - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # ==========================
    # PANEL DE ESTADO (Lateral)
    # ==========================
    y_start = 30
    color_texto_info = (220, 220, 220)

    # Audio Info
    audio_str = f"Audio: {audio_stats['status']} ({audio_stats['rms']:.3f})"
    cv2.putText(frame, audio_str, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto_info, 1)

    # Movimiento Info
    move_str = f"Activ: {move_status}"
    cv2.putText(frame, move_str, (10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto_info, 1)

    # Estado de Ánimo (Grande)
    color_mood = (0, 255, 0) # Verde (Normal)
    if mood == "Desesperado":
        color_mood = (0, 0, 255) # Rojo
    elif mood == "Estresado":
        color_mood = (0, 165, 255) # Naranja
    elif mood == "Comodo":
        color_mood = (255, 255, 0) # Cyan/Amarillo

    cv2.putText(frame, f"Estado: {mood.upper()}", (10, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mood, 2)

    # Alertas de Salud
    if health_alerts:
        y_alert = y_start + 100
        for alert in health_alerts:
            cv2.putText(frame, alert, (10, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_alert += 25

    # ==========================
    # BARRA SUPERIOR (Acción)
    # ==========================
    if accion:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (ancho, 60), (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        texto = f"!! {accion.upper()} !!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (text_w, _text_h), _ = cv2.getTextSize(texto, font, scale, thickness)
        text_x = (ancho - text_w) // 2

        color_texto = (0, 255, 255)
        if "Alimentacion" in accion:
            color_texto = (0, 0, 255)
        elif "Hidratacion" in accion:
            color_texto = (255, 0, 0)

        cv2.putText(frame, texto, (text_x, 40), font, scale, color_texto, thickness)


def iniciar_monitoreo():
    print("🧠 Cargando IA...")
    try:
        model = YOLO("best.pt")
    except Exception:
        print("❌ ERROR: No se encuentra 'best.pt'")
        return

    # Tracker config
    TRACKER_CONFIG = "botsort.yaml"

    url_tapo = _obtener_url_rtsp_tapo()
    if not url_tapo:
        return

    # --- INICIALIZAR MÓDULOS INTELIGENTES ---
    print("🎧 Iniciando Monitor de Audio...")
    audio_mon = AudioMonitor(url_tapo)
    audio_mon.start()

    print("🐾 Iniciando Monitor de Movimiento...")
    move_mon = MovementMonitor()

    print("❤️ Iniciando Monitor de Salud...")
    health_mon = HealthMonitor()
    # -----------------------------------------

    print("🎥 Conectando cámara...")
    cap = cv2.VideoCapture(url_tapo)

    AREA_MINIMA = 1200
    AREA_MAXIMA = 85000
    CONF_NINFA = 0.20
    CONF_PLATOS = 0.35

    HOLD_ACCION_SEGUNDOS = 3.0
    MIN_DURACION_PARA_GUARDAR = 2.0

    memoria_cajas = []
    accion_estable = ""
    accion_ultima_vez_vista = 0.0
    accion_inicio_ts = 0.0
    accion_inicio_logueado = False
    conf_accion_ultima = 0.0

    frame_count = 0
    SKIP_FRAMES = 3

    prev_time = time.time()
    fps = 0.0

    # Tracking BoT-SORT
    importlib.invalidate_caches()
    tracking_activo = importlib.util.find_spec("lap") is not None
    aviso_tracking_fallido = False
    aviso_tracking_reactivado = False
    model_det = model

    print("🦅 MONITOR PRO ACTIVADO: Sistema Multimodal 🦅")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Señal perdida. Reconectando...")
                time.sleep(2)
                cap.open(url_tapo)
                continue

            ahora_frame = time.time()
            fps = 1 / (ahora_frame - prev_time) if (ahora_frame - prev_time) > 0 else 0
            prev_time = ahora_frame

            # =========================================================
            #  IA - Cada X frames
            # =========================================================
            frame_count += 1
            if frame_count % SKIP_FRAMES == 0:
                try:
                    if tracking_activo:
                        results = model.track(
                            frame,
                            persist=True,
                            verbose=False,
                            conf=0.15,
                            iou=0.5,
                            tracker=TRACKER_CONFIG,
                        )
                    else:
                        results = model_det(frame, verbose=False, conf=0.15, iou=0.5)
                except Exception as e:
                    if not aviso_tracking_fallido:
                        print(f"⚠️ Fallo en tracker, usando modo simple: {e}")
                        aviso_tracking_fallido = True
                    tracking_activo = False

                    try:
                        model.predictor = None
                    except Exception:
                        pass

                    if model_det is model:
                        try:
                            model_det = YOLO("best.pt")
                        except Exception:
                            model_det = model

                    results = model_det(frame, verbose=False, conf=0.15, iou=0.5)

                if not tracking_activo:
                    importlib.invalidate_caches()
                    if importlib.util.find_spec("lap") is not None:
                        tracking_activo = True
                        if not aviso_tracking_reactivado:
                            print("✅ 'lap' detectado. Reactivando tracking BoT-SORT...")
                            aviso_tracking_reactivado = True

                temp_comida = []
                temp_agua = []
                temp_ninfa = []
                memoria_cajas = []
                accion_detectada_hoy = ""
                max_conf_frame = 0.0

                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        nombre = model.names[cls_id]
                        coords = list(map(int, box.xyxy[0]))
                        conf = float(box.conf[0])

                        track_id = None
                        try:
                            if box.id is not None:
                                track_id = int(box.id[0])
                        except Exception:
                            track_id = None

                        ancho = coords[2] - coords[0]
                        alto = coords[3] - coords[1]
                        area = ancho * alto

                        if area < AREA_MINIMA or area > AREA_MAXIMA:
                            continue

                        color = (0, 255, 0)
                        if nombre == "ninfa":
                            if conf < CONF_NINFA:
                                continue
                            temp_ninfa.append(coords)
                            max_conf_frame = max(max_conf_frame, conf)

                            # --- UPDATE MOVIMIENTO ---
                            cx = (coords[0] + coords[2]) // 2
                            cy = (coords[1] + coords[3]) // 2
                            if track_id is not None:
                                move_mon.update(track_id, cx, cy)
                            # -------------------------

                        else:
                            if conf < CONF_PLATOS:
                                continue
                            if nombre == "comedero":
                                temp_comida.append(coords)
                                color = (0, 0, 255)
                            elif nombre == "bebedero":
                                temp_agua.append(coords)
                                color = (255, 0, 0)

                        memoria_cajas.append((coords, nombre, color, area, track_id))

                for nx1, ny1, nx2, ny2 in temp_ninfa:
                    cx, cy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
                    for x1, y1, x2, y2 in temp_comida:
                        if (x1 < cx < x2) and (y1 < cy < y2):
                            accion_detectada_hoy = "Alimentacion"
                    for x1, y1, x2, y2 in temp_agua:
                        if (x1 < cx < x2) and (y1 < cy < y2):
                            accion_detectada_hoy = "Hidratacion"

                if accion_detectada_hoy:
                    accion_ultima_vez_vista = ahora_frame
                    conf_accion_ultima = max_conf_frame

                    if accion_estable != accion_detectada_hoy:
                        if accion_estable and accion_inicio_logueado:
                            duracion = max(0.0, ahora_frame - accion_inicio_ts)
                            if duracion >= MIN_DURACION_PARA_GUARDAR:
                                print(f"🔀 CAMBIO ACCIÓN: {accion_estable} -> {accion_detectada_hoy}")
                                guardar_async(
                                    "Accion",
                                    accion_estable,
                                    float(duracion),
                                    "Fin",
                                    float(conf_accion_ultima),
                                    "Cambio",
                                )

                        accion_estable = accion_detectada_hoy
                        accion_inicio_ts = ahora_frame
                        accion_inicio_logueado = False

            # =========================================================
            #  LÓGICA DE ESTADO (Continuo)
            # =========================================================

            # 1) Expiración (FIN)
            if accion_estable and (ahora_frame - accion_ultima_vez_vista) > HOLD_ACCION_SEGUNDOS:
                accion_prev = accion_estable
                accion_estable = ""

                duracion = max(0.0, accion_ultima_vez_vista - accion_inicio_ts)
                if accion_inicio_logueado and duracion >= MIN_DURACION_PARA_GUARDAR:
                    print(f"⏹️ FIN ACCIÓN: {accion_prev} ({duracion:.1f}s)")
                    guardar_async(
                        "Accion",
                        accion_prev,
                        float(duracion),
                        "Fin",
                        float(conf_accion_ultima),
                        f"Duracion={duracion:.1f}s | TapoMovil",
                    )

                accion_inicio_logueado = False
                accion_inicio_ts = 0.0

            # 2) Confirmación (INICIO válido)
            if accion_estable and (not accion_inicio_logueado) and accion_inicio_ts:
                if (ahora_frame - accion_inicio_ts) >= MIN_DURACION_PARA_GUARDAR:
                    print(f"▶️ INICIO ACCIÓN: {accion_estable}")

                    # --- Registrar Salud ---
                    health_mon.register_action(accion_estable)
                    # -----------------------

                    guardar_async(
                        "Accion",
                        accion_estable,
                        1.0,
                        "Inicio",
                        float(conf_accion_ultima),
                        f"Min={MIN_DURACION_PARA_GUARDAR:.0f}s | TapoMovil",
                    )
                    accion_inicio_logueado = True

            # =========================================================
            #  DETERMINAR ESTADOS (MOOD/HEALTH)
            # =========================================================
            audio_stats = audio_mon.get_status()
            move_status, move_val = move_mon.get_global_activity()
            health_alerts = health_mon.check_health()

            # Lógica de Estado de Ánimo (Mood)
            mood = "Normal"
            is_screaming = audio_stats["status"] == "Gritando"
            is_high_activity = "Alta" in move_status or "Muy Activo" in move_status

            # Prioridad de Estados
            if len(health_alerts) > 0 and is_screaming:
                mood = "Desesperado"
                # Podríamos guardar log aquí si cambia el estado
            elif is_screaming and is_high_activity:
                mood = "Estresado"
            elif move_status == "Tranquilo" and audio_stats["status"] in ["Sonidos", "Silencio"]:
                mood = "Comodo"

            # =========================================================
            #  DIBUJO
            # =========================================================
            for coords, nombre, color, area, track_id in memoria_cajas:
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                etiqueta = nombre
                if track_id is not None:
                    etiqueta += f" #{track_id}"
                cv2.putText(frame, etiqueta, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            dibujar_hud(frame, accion_estable, fps, audio_stats, move_status, mood, health_alerts)

            cv2.imshow("Monitor Ninfas Pro", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cerrar todo limpiamente
        print("🛑 Deteniendo monitores...")
        audio_mon.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_monitoreo()