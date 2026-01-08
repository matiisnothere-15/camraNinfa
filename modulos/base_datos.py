import pyodbc
from config.credenciales import AZURE_CONFIG

def obtener_conexion():
    """Crea y devuelve la conexión a Azure"""
    try:
        conn_str = (
            f"DRIVER={AZURE_CONFIG['driver']};"
            f"SERVER={AZURE_CONFIG['server']};"
            f"PORT=1433;"
            f"DATABASE={AZURE_CONFIG['database']};"
            f"UID={AZURE_CONFIG['username']};"
            f"PWD={AZURE_CONFIG['password']};"
            "Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"❌ Error crítico de conexión: {e}")
        return None

def insertar_registro(categoria, accion, valor, estado, confianza, notas=""):
    """Guarda un evento en la tabla BitacoraAves"""
    conn = obtener_conexion()
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO BitacoraAves 
            (Categoria, Accion, Valor_Numerico, Estado_Observado, Confianza_IA, Notas)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (categoria, accion, valor, estado, confianza, notas))
        conn.commit()
        print(f"☁️ [AZURE] Registro guardado: {accion}")
        return True
    except Exception as e:
        print(f"⚠️ Error al insertar: {e}")
        return False
    finally:
        conn.close()