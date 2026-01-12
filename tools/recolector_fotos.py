import cv2
import time
import os
import uuid

# Configuración
CARPETA_DESTINO = "dataset/images"
INTERVALO_SEGUNDOS = 5  # Saca una foto cada 5 segundos
TOTAL_FOTOS = 50        # Cuántas fotos quieres sacar en esta sesión

# Crear carpeta si no existe
if not os.path.exists(CARPETA_DESTINO):
    os.makedirs(CARPETA_DESTINO)

print(f"📸 INICIANDO RECOLECCIÓN DE DATASET (DroidCam)")
print(f"   Destino: {CARPETA_DESTINO}")
print(f"   Intervalo: {INTERVALO_SEGUNDOS} segundos")
print("---------------------------------------")

# --- CAMBIO REALIZADO AQUÍ ---
# Conexión directa a tu celular (IP 192.168.1.172)
print("🎥 Conectando al celular...")
cap = cv2.VideoCapture("http://192.168.0.6:4747/mjpegfeed?640x480")

if not cap.isOpened():
    print("❌ Error: No se pudo conectar a DroidCam.")
    print("   -> Revisa que la app esté abierta y el celular desbloqueado.")
    exit()

contador = 0
ultimo_tiempo = time.time()

while contador < TOTAL_FOTOS:
    ret, frame = cap.read()
    if not ret: 
        print("⚠️ Error leyendo frame (pantalla negra o desconexión).")
        break

    tiempo_actual = time.time()
    
    # Cuenta regresiva visual en pantalla
    tiempo_restante = INTERVALO_SEGUNDOS - (tiempo_actual - ultimo_tiempo)
    
    # Hacemos una COPIA del frame para dibujar los textos
    # (Así la foto que se guarda sale limpia, sin letras encima)
    frame_preview = frame.copy()

    cv2.putText(frame_preview, f"Foto {contador+1}/{TOTAL_FOTOS}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_preview, f"Prox: {tiempo_restante:.1f}s", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Lógica de captura
    if tiempo_restante <= 0:
        # Generar nombre único para la foto
        nombre_foto = f"ninfa_{uuid.uuid4().hex[:8]}.jpg"
        ruta_completa = os.path.join(CARPETA_DESTINO, nombre_foto)
        
        # Guardamos el frame ORIGINAL (limpio, sin texto)
        cv2.imwrite(ruta_completa, frame)
        print(f"💾 Guardada ({contador+1}/{TOTAL_FOTOS}): {nombre_foto}")
        
        contador += 1
        ultimo_tiempo = time.time()

    # Mostrar lo que ve la cámara
    cv2.imshow("Recolector de Dataset", frame_preview)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ ¡Sesión de fotos terminada!")