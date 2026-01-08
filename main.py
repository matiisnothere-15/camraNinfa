from modulos.vision import iniciar_monitoreo

def iniciar_sistema():
    print("---------------------------------------")
    print("🦅 SISTEMA MONITOR DE NINFAS - INICIANDO")
    print("   Arquitectura Modular: [OK]")
    print("---------------------------------------")
    
    # Arrancar el módulo de visión
    # Este módulo ya sabe cómo hablar con la base de datos.
    iniciar_monitoreo()

if __name__ == "__main__":
    iniciar_sistema()