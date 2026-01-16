import os

import pyodbc


def _cargar_config_azure() -> dict:
    """Carga la configuración de Azure SQL.

    Prioridad:
    1) Variables de entorno (recomendado)
    2) config.credenciales.AZURE_CONFIG (si existe localmente)
    """

    server = os.getenv("AZURE_SERVER")
    database = os.getenv("AZURE_DATABASE")
    username = os.getenv("AZURE_USERNAME")
    password = os.getenv("AZURE_PASSWORD")
    driver = os.getenv("AZURE_DRIVER", "{ODBC Driver 18 for SQL Server}")

    if server and database and username and password:
        return {
            "server": server,
            "database": database,
            "username": username,
            "password": password,
            "driver": driver,
        }

    try:
        from config.credenciales import AZURE_CONFIG  # type: ignore

        return AZURE_CONFIG
    except Exception:
        return {}

def obtener_conexion():
    """Crea y devuelve la conexión a Azure"""
    config = _cargar_config_azure()
    if not config:
        print(
            "❌ Azure no configurado. Define AZURE_SERVER, AZURE_DATABASE, AZURE_USERNAME, AZURE_PASSWORD (y opcional AZURE_DRIVER)."
        )
        return None

    try:
        conn_str = (
            f"DRIVER={config['driver']};"
            f"SERVER={config['server']};"
            f"PORT=1433;"
            f"DATABASE={config['database']};"
            f"UID={config['username']};"
            f"PWD={config['password']};"
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