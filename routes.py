import io
import pandas as pd
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from src import utils
from src import globals 

# Crear el enrutador de la API
router = APIRouter()

# Clase para solicitudes con URL de origen
class SourceURLRequest(BaseModel):
    pdf_url: str

# Clase para solicitudes de consulta
class QueryRequest(BaseModel):
    query: str

# Ruta para cargar fuente desde una URL
@router.post("/source_url")
def load_source_from_url(request: SourceURLRequest):
    pdf_url = request.pdf_url
    globals.logger.info(f"Recibida solicitud para cargar fuente desde URL: {pdf_url}")
    try:
        # Descargar el contenido del PDF
        pdf_content = utils.download_pdf(pdf_url, globals.logger)
        # Procesar el PDF e insertar embeddings
        utils.process_pdf_and_insert_embeddings(
            pdf_content, globals.model, globals.device, globals.database, globals.table, globals.logger
        )
        globals.source_loaded = True
        return {"message": "Fuente cargada y procesada exitosamente desde URL."}
    except Exception as e:
        globals.logger.error(f"Error al procesar la fuente: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para cargar fuente desde un archivo
@router.post("/source_file")
def load_source_from_file(file: UploadFile = File(...)):
    globals.logger.info("Recibida solicitud para cargar fuente desde archivo.")
    try:
        # Leer los bytes del archivo PDF
        pdf_bytes = file.file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="No se proporcionó contenido de archivo.")
        pdf_content = io.BytesIO(pdf_bytes)
        # Procesar el PDF e insertar embeddings
        utils.process_pdf_and_insert_embeddings(
            pdf_content, globals.model, globals.device, globals.database, globals.table, globals.logger
        )
        globals.source_loaded = True
        return {"message": "Fuente cargada y procesada exitosamente desde archivo."}
    except Exception as e:
        globals.logger.error(f"Error al procesar la fuente desde archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para manejar consultas
@router.post("/query")
def handle_query(request: QueryRequest):
    query_text = request.query
    globals.logger.info(f"Consulta recibida: {query_text}")

    if not globals.source_loaded:
        return {"message": "No se ha cargado ninguna fuente aún."}

    try:
        # Convertir la consulta a un vector de embedding
        query_vector = utils.queryToEmbedding(query_text, globals.model, globals.device)

        index_name = "flat_index"
        # Buscar resultados similares en imágenes
        image_results = globals.table.search(
            vectors={index_name: [query_vector.tolist()]},
            n=3,
            filter=[("like", "media_type", "image")]
        )
        # Buscar resultados similares en texto
        text_results = globals.table.search(
            vectors={index_name: [query_vector.tolist()]},
            n=3,
            filter=[("like", "media_type", "text")]
        )

        # Combinar los resultados
        results = [pd.concat([image_results[0], text_results[0]], ignore_index=True)]
        globals.logger.info("Búsqueda de similitud completada.")

        # Preparar los datos recuperados para RAG y las imágenes en base64
        retrieved_data_for_RAG, images_base64 = utils.RAG_Setup(results, [query_text], globals.logger)
        globals.logger.info("Datos recuperados preparados para generar respuesta.")

        # Generar la respuesta usando el modelo de visión
        response = globals.vision_model.generate_content(retrieved_data_for_RAG)
        globals.logger.info("Respuesta generada exitosamente.")
        return {
            "response": response.text,
            "images": images_base64[0] if images_base64 else None
        }
    except Exception as e:
        globals.logger.error(f"Error al manejar la consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))