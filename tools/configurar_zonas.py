import cv2
import numpy as np

# Variable para guardar los puntos que clickeas
puntos_comedero = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"📍 Punto capturado: {x}, {y}")
        puntos_comedero.append([x, y])
        
        # Dibujar un círculo donde hiciste clic
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Configurador de Zonas", img)

print("-------------------------------------------------")
print("🛠️ HERRAMIENTA DE CONFIGURACIÓN DE ZONAS")
print("1. Se abrirá la cámara.")
print("2. Haz CLIC en las 4 esquinas del COMEDERO.")
print("3. Cuando tengas los 4 puntos rojos, presiona 'q'.")
print("-------------------------------------------------")

cap = cv2.VideoCapture(0) # Usa 0 por ahora (Webcam)

ret, img = cap.read()
if not ret:
    print("❌ Error al leer cámara")
    exit()

cv2.imshow("Configurador de Zonas", img)
cv2.setMouseCallback("Configurador de Zonas", click_event)

# Esperar hasta que presiones 'q'
while True:
    cv2.imshow("Configurador de Zonas", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n\n✅ ¡LISTO! COPIA ESTOS NÚMEROS EN TU CÓDIGO:")
print(f"zona_comedero = np.array({puntos_comedero}, np.int32)")
print("-------------------------------------------------")