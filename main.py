import cv2
import time
import numpy as np 
from ultralytics import YOLO
from modulos.base_datos import insertar_registro 

def iniciar_monitoreo():
    # ---------------------------------------------------------
    # 1. CARGA DE CEREBRO
    # ---------------------------------------------------------
    print("🧠 Cargando Modelo (best.pt)...")
    try:
        model = YOLO('best.pt') 
        print("✅ Modelo cargado.")
    except:
        print("❌ ERROR: No encuentro 'best.pt'.")
        return

    # ---------------------------------------------------------
    # 2. CONFIGURACIÓN DE CÁMARAS
    # ---------------------------------------------------------
    # IPS de tus cámaras (Revisar si cambiaron)
    url_lado = "http://192.168.0.6:4747/mjpegfeed?640x480"   
    url_techo = "http://192.168.0.10:4747/mjpegfeed?640x480" 

    print(f"🎥 Conectando Cámaras...")
    cap_lado = cv2.VideoCapture(url_lado)
    cap_techo = cv2.VideoCapture(url_techo)

    # Variables de control
    ultimo_registro = 0
    TIEMPO_ESPERA = 5 
    SKIP_FRAMES = 3   
    frame_count = 0

    # === MEMORIA VISUAL (Para evitar parpadeo) ===
    # Aquí guardaremos las últimas cajas detectadas para dibujarlas siempre
    memoria_lado = []
    memoria_techo_cajas = [] # Guardará tuplas (x1, y1, x2, y2, clase, color)
    memoria_accion = ""      # Guardará el texto de "COMIENDO"

    print("🦅 SISTEMA ESTABILIZADO INICIADO 🦅")
    
    while True:
        ret1, frame_lado = cap_lado.read()
        ret2, frame_techo = cap_techo.read()

        if not ret1: frame_lado = np.zeros((480, 640, 3), dtype=np.uint8)
        if not ret2: frame_techo = np.zeros((480, 640, 3), dtype=np.uint8)

        # Resize preventivo
        try:
            frame_lado = cv2.resize(frame_lado, (640, 480))
            frame_techo = cv2.resize(frame_techo, (640, 480))
        except: pass

        frame_count += 1
        
        # -----------------------------------------------------------
        # FASE 1: DETECCIÓN (Solo ocurre cada X cuadros)
        # -----------------------------------------------------------
        if frame_count % SKIP_FRAMES == 0:
            
            # --- CÁMARA LADO ---
            res_lado = model(frame_lado, verbose=False, conf=0.25)
            memoria_lado = [] # Limpiamos memoria vieja
            for r in res_lado:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    nombre = model.names[cls_id]
                    if nombre == 'ninfa':
                        coords = list(map(int, box.xyxy[0]))
                        confianza = float(box.conf[0])
                        memoria_lado.append((coords, confianza))

            # --- CÁMARA TECHO ---
            res_techo = model(frame_techo, verbose=False, conf=0.25)
            
            # Listas temporales para lógica
            temp_comida = []
            temp_agua = []
            temp_ninfa = []
            
            memoria_techo_cajas = [] # Limpiamos memoria vieja
            memoria_accion = ""      # Reseteamos acción

            for r in res_techo:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    nombre = model.names[cls_id] 
                    coords = list(map(int, box.xyxy[0]))
                    
                    # Guardamos para lógica
                    if nombre == 'comedero': temp_comida.append(coords)
                    elif nombre == 'bebedero': temp_agua.append(coords)
                    elif nombre == 'ninfa': temp_ninfa.append(coords)
                    
                    # Asignar color y guardar en memoria visual
                    color = (255, 255, 255)
                    if nombre == 'comedero': color = (0, 0, 255) # Rojo
                    elif nombre == 'bebedero': color = (255, 0, 0) # Azul
                    elif nombre == 'ninfa': color = (0, 255, 0) # Verde
                    
                    # Guardamos TODO lo necesario para dibujar después
                    memoria_techo_cajas.append((coords, nombre, color))

            # --- LÓGICA DE NEGOCIO ---
            for nx1, ny1, nx2, ny2 in temp_ninfa:
                centro_x = (nx1 + nx2) // 2
                centro_y = (ny1 + ny2) // 2
                
                # Chequeo Comida
                for cx1, cy1, cx2, cy2 in temp_comida:
                    if (cx1 < centro_x < cx2) and (cy1 < centro_y < cy2):
                        memoria_accion = "Alimentacion" # Guardamos en memoria

                # Chequeo Agua
                for ax1, ay1, ax2, ay2 in temp_agua:
                    if (ax1 < centro_x < ax2) and (ay1 < centro_y < ay2):
                        memoria_accion = "Hidratacion" # Guardamos en memoria

            # Guardar en Azure (Solo si hay acción nueva)
            tiempo_actual = time.time()
            if memoria_accion and (tiempo_actual - ultimo_registro > TIEMPO_ESPERA):
                print(f"🚀 ENVIANDO A AZURE: {memoria_accion}")
                try:
                    insertar_registro("Comportamiento", memoria_accion, 1, "Activo", 0.95, "CamaraTecho")
                    ultimo_registro = tiempo_actual
                except Exception as e:
                    print(f"Error Azure: {e}")

        # -----------------------------------------------------------
        # FASE 2: DIBUJO (Ocurre SIEMPRE usando la memoria)
        # -----------------------------------------------------------
        
        # Dibujar Lado
        for coords, conf in memoria_lado:
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame_lado, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_lado, f"Ninfa {conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Dibujar Techo
        for coords, nombre, color in memoria_techo_cajas:
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame_techo, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_techo, nombre, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Dibujar Acción (Si la memoria dice que están comiendo)
        if memoria_accion == "Alimentacion":
            cv2.putText(frame_techo, "!! COMIENDO !!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif memoria_accion == "Hidratacion":
            cv2.putText(frame_techo, "!! BEBIENDO !!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Mostrar
        pantalla_final = np.hstack((frame_lado, frame_techo))
        cv2.imshow("Monitor Tesis - Estabilizado", cv2.resize(pantalla_final, (1000, 375)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_lado.release()
    cap_techo.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_monitoreo()