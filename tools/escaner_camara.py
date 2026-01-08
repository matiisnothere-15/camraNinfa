import cv2

print("📱 PROBANDO CONEXIÓN CON DROIDCAM")
print("--------------------------------")

# URL sacada de tu captura de pantalla
# DroidCam usa HTTP, no RTSP
url_droidcam = "http://192.168.1.172:4747/video"

print(f"📡 Conectando a: {url_droidcam}")
cap = cv2.VideoCapture(url_droidcam)

if not cap.isOpened():
    print("❌ ERROR: No conecta.")
    print("Tips:")
    print("1. Asegúrate de que DroidCam siga abierto en el celular.")
    print("2. Que el celular no se haya bloqueado.")
    print("3. Que PC y Celular sigan en el mismo WiFi.")
else:
    print("✅ ¡CONEXIÓN EXITOSA! (Presiona 'q' para salir)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Se perdió la señal.")
            break
        
        # Reducimos un poco si se ve muy gigante
        frame = cv2.resize(frame, (640, 480))
        
        cv2.imshow("Prueba DroidCam - Python", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()