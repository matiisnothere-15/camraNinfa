import av
import numpy as np
import threading
import time

class AudioMonitor:
    def __init__(self, rtsp_url):
        self.url = rtsp_url
        self.running = False
        self.thread = None
        self.latest_status = "Silencio"
        self.latest_rms = 0.0
        self.latest_freq = 0.0
        self.lock = threading.Lock()

        # Configuración de umbrales
        # Estos valores necesitan calibración en el entorno real
        self.RMS_THRESHOLD_SCREAM = 0.2  # Volumen alto
        self.RMS_THRESHOLD_CONTENT = 0.02 # Volumen medio/bajo
        self.FREQ_THRESHOLD_HIGH = 1500 # Hz, umbral para chillidos agudos

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def get_status(self):
        with self.lock:
            return {
                "status": self.latest_status,
                "rms": self.latest_rms,
                "freq": self.latest_freq
            }

    def _monitor_loop(self):
        print(f"🎤 Iniciando monitoreo de audio en hilo secundario...")
        while self.running:
            container = None
            try:
                # Opciones para reducir latencia en la conexión
                options = {'rtsp_transport': 'tcp', 'stimeout': '5000000'}
                container = av.open(self.url, options=options)

                # Buscar stream de audio
                if not container.streams.audio:
                    print("⚠️ No se encontró stream de audio.")
                    time.sleep(5)
                    continue

                stream = container.streams.audio[0]

                # Resampler para estandarizar a mono, 16kHz
                resampler = av.AudioResampler(format='flt', layout='mono', rate=16000)

                for frame in container.decode(stream):
                    if not self.running:
                        break

                    # Procesar frame
                    out_frames = resampler.resample(frame)

                    for out_frame in out_frames:
                        audio_data = out_frame.to_ndarray()
                        # Procesar en chunks
                        self._analyze_chunk(audio_data)

            except Exception as e:
                # print(f"⚠️ Error audio loop: {e}. Reintentando en 5s...")
                # Evitar spam de logs si falla constantemente
                time.sleep(5)
            finally:
                if container:
                    try:
                        container.close()
                    except:
                        pass

    def _analyze_chunk(self, data):
        if len(data) == 0:
            return

        # Calcular RMS (Volumen)
        rms = np.sqrt(np.mean(data**2))

        # Calcular Frecuencia Dominante (FFT)
        freq = 0.0
        if rms > 0.001:
            # Hanning window para suavizar bordes
            windowed = data * np.hanning(len(data))
            w = np.fft.fft(windowed)
            freqs = np.fft.fftfreq(len(w))
            idx = np.argmax(np.abs(w))
            freq = freqs[idx] * 16000 # Tasa de muestreo
            freq = abs(freq)

        status = "Silencio"

        # Lógica heurística simple
        if rms > self.RMS_THRESHOLD_SCREAM:
            if freq > self.FREQ_THRESHOLD_HIGH:
                status = "Gritando" # Agudo y fuerte
            else:
                status = "Ruido Fuerte"
        elif rms > self.RMS_THRESHOLD_CONTENT:
            # Si el volumen es moderado
            status = "Sonidos"
            # Aquí podríamos diferenciar "Contento" si tuviéramos más métricas
            # Por ahora asumimos que sonidos suaves son positivos o neutros

        with self.lock:
            self.latest_rms = float(rms)
            self.latest_freq = float(freq)
            self.latest_status = status
