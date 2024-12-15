# Importar las bibliotecas necesarias
import sys
import os
import logging
import warnings
import torch
import kdbai_client as kdbai
import google.generativeai as genai
from fastapi import FastAPI
from dotenv import load_dotenv
from imagebind.models import imagebind_model
from src.routes import router as routes_router
from src import globals  

# Cargar variables de entorno
load_dotenv()
# Ignorar advertencias
warnings.filterwarnings('ignore')

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)
# Configurar el logger global
globals.logger = logging.getLogger(__name__)

# Inicializar la aplicación FastAPI
app = FastAPI()
globals.logger.info("Aplicación FastAPI inicializada.")

# Configurar el dispositivo (GPU si está disponible)
globals.device = "cuda:0" if torch.cuda.is_available() else "cpu"
globals.logger.info(f"Usando dispositivo: {globals.device}")
# Cargar y preparar el modelo ImageBind
globals.model = imagebind_model.imagebind_huge(pretrained=True)
globals.model.eval()
globals.model.to(globals.device)
globals.logger.info("Modelo ImageBind cargado y movido al dispositivo.")

# Obtener claves de API y endpoints de variables de entorno
KDBAI_API_KEY = os.getenv('KDBAI_API_KEY')
KDBAI_ENDPOINT = os.getenv('KDBAI_ENDPOINT')

# Verificar que las variables de entorno estén configuradas
if not KDBAI_ENDPOINT or not KDBAI_API_KEY:
    globals.logger.error("Las variables de entorno KDBAI_ENDPOINT y KDBAI_API_KEY deben estar configuradas.")
    sys.exit(1)

# Inicializar sesión con KDB.ai
globals.logger.info("Inicializando sesión con KDB.ai.")
session = kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)
globals.database = session.database('default')

# Definir el esquema y el índice de la tabla
schema = [
    {"name": "path", "type": "str"},
    {"name": "media_type", "type": "str"},
    {"name": "embeddings", "type": "float32s"},
    {"name": "page_number", "type": "int64"}
]

index = [
    {
        "name": "flat_index",
        "type": "flat",
        "column": "embeddings",
        "params": {"dims": 1024, "metric": "CS"},
    }
]

table_name = 'multimodalImageBind'

# Verificar si la tabla ya existe en la base de datos
try:
    globals.table = globals.database.table(table_name)
    globals.logger.info(f"Table '{table_name}' found. Using existing table.")
except kdbai.KDBAIException:
    globals.logger.info(f"Table '{table_name}' not found. Creating a new one.")
    globals.table = globals.database.create_table(table_name, schema=schema, indexes=index)

# Verificar si hay embeddings existentes en la tabla
df_all = globals.table.query()

# Check how many rows are returned
count = len(df_all)
if count > 0:
    globals.source_loaded = True
    globals.logger.info("Existing embeddings found. Set source_loaded to True.")
else:
    globals.source_loaded = False
    globals.logger.info("No existing embeddings found. Set source_loaded to False.")

# Obtener la clave de API de Google
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    globals.logger.error("La variable de entorno GOOGLE_API_KEY debe estar configurada.")
    sys.exit(1)

# Configurar el modelo de Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
globals.vision_model = genai.GenerativeModel('gemini-1.5-flash')
globals.logger.info("Modelo de Google Generative AI configurado.")

# Incluir las rutas en la aplicación
app.include_router(routes_router)