import cv2
import time
from ultralytics import YOLO  # type: ignore
from modulos.base_datos import insertar_registro

def iniciar_monitoreo():
    print("🧠 Cargando modelo IA (YOLOv8)...")
    model = YOLO('yolov8n.pt') 

    # --- CAMBIO IMPORTANTE: Usamos '/mjpegfeed' en vez de '/video' ---
    url_droidcam = "http://192.168.0.6:4747/mjpegfeed?640x480"
    print(f"🎥 Conectando a DroidCam: {url_droidcam}")
    
    cap = cv2.VideoCapture(url_droidcam)

    if not cap.isOpened():
        print("❌ Error CRÍTICO: Python no pudo abrir la conexión.")
        print("   -> Verifica que la IP sea 192.168.1.172")
        return

    print("🦅 VIGILANCIA ACTIVA. Esperando primer frame...")
    
    # Variables de control
    ultimo_registro = 0
    TIEMPO_ESPERA = 15
    frames_leidos = 0

    while True:
        ret, frame = cap.read()
        
        # Diagnóstico de video
        if not ret:
            print("⚠️ Conectado, pero NO LLEGA IMAGEN (Frame vacío).")
            print("   -> Intenta cerrar y abrir la app DroidCam en el celular.")
            break
        
        frames_leidos += 1
        if frames_leidos == 1:
            print("✅ ¡IMAGEN RECIBIDA! Abriendo ventana...")

        # 1. Detección
        resultados = model(frame, classes=[14], verbose=False)
        
        pajaro_detectado = False
        confianza_actual = 0.0

        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > 0.50:
                    pajaro_detectado = True
                    confianza_actual = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Ninfa {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Guardar en Azure
        tiempo_actual = time.time()
        if pajaro_detectado and (tiempo_actual - ultimo_registro > TIEMPO_ESPERA):
            print(f"👀 ¡Ninfa vista! ({confianza_actual:.2f}) -> Guardando...")
            exito = insertar_registro("Observacion", "Presencia Detectada", 1, "Activo", confianza_actual, "DroidCam")
            if exito: ultimo_registro = tiempo_actual

        # 3. Mostrar Video
        # Forzamos un tamaño estándar para asegurar que se vea bien
        frame_show = cv2.resize(frame, (800, 600))
        cv2.imshow("Monitor Ninfas (DroidCam)", frame_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()