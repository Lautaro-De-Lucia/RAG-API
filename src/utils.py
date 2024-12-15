import os
import io
import uuid
import base64
import requests
import fitz
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from fastapi import HTTPException
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
import torch

# Función para descargar un PDF desde una URL o archivo local
def download_pdf(source, logger):
    logger.info(f"Intentando descargar PDF desde la fuente: '{source}'")
    if source.lower().startswith(('http://', 'https://')):
        logger.info("La fuente es una URL.")
        response = requests.get(source)
        if response.status_code != 200:
            logger.error(f"No se pudo descargar el PDF desde la URL. Código de estado: {response.status_code}")
            raise HTTPException(status_code=400, detail="No se pudo descargar el PDF desde la URL proporcionada.")
        logger.info("PDF descargado exitosamente desde la URL.")
        return io.BytesIO(response.content)
    elif os.path.isfile(source):
        logger.info("La fuente es un archivo local.")
        try:
            with open(source, 'rb') as file:
                logger.info("Archivo PDF abierto exitosamente.")
                return io.BytesIO(file.read())
        except Exception as e:
            logger.error(f"Error al leer el archivo PDF: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        logger.error(f"La fuente no es una URL válida ni un archivo: {source}")
        raise HTTPException(status_code=400, detail="Fuente inválida. No es una URL ni un archivo.")

# Función para extraer contenidos del PDF
def extract_pdf_contents(pdf_content, logger):
    images_dir = "images"
    texts_dir = "texts"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    texts = []
    image_paths = []

    logger.info("Iniciando extracción de contenidos del PDF.")
    for page_num in tqdm(range(len(doc)), desc="Procesando páginas"):
        page = doc[page_num]
        # Extraer texto de la página
        text = page.get_text()
        if text.strip():
            text_filename = f"page_{page_num + 1}.txt"
            text_path = os.path.join(texts_dir, text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            texts.append(text_path)
            logger.debug(f"Texto extraído de la página {page_num + 1}.")

        # Extraer imágenes de la página
        image_list = page.get_images(full=True)
        for _, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(images_dir, image_filename)

            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_path)
            image_paths.append(image_path)
            logger.debug(f"Imagen {image_filename} extraída de la página {page_num + 1}.")

    doc.close()
    logger.info("Extracción de contenidos del PDF finalizada.")
    return texts, image_paths

# Función para leer texto desde un archivo
def read_text_from_file(filename, logger=None):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except IOError as e:
        if logger:
            logger.error(f"Error al leer el archivo {filename}: {e}")
        return None

# Función para obtener el vector de embedding
def getEmbeddingVector(inputs, model):
    with torch.no_grad():
        embedding = model(inputs)
    for _, value in embedding.items():
        vec = value.reshape(-1)
        vec = vec.cpu().numpy()
        return vec

# Función para convertir datos a embeddings
def dataToEmbedding(dataIn, dtype, model, device):
    if dtype == 'image':
        data_path = [dataIn]
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(data_path, device)
        }
    elif dtype == 'text':
        txt = [dataIn]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(txt, device)
        }
    vec = getEmbeddingVector(inputs, model)
    return vec

# Función para crear un dataframe de embeddings
def create_embeddings_dataframe(texts, image_paths, model, device, logger):
    columns = ['path', 'media_type', 'embeddings', 'page_number']
    df = pd.DataFrame(columns=columns)

    logger.info("Generando embeddings para datos de texto.")
    for text_path in tqdm(texts, desc="Procesando textos"):
        text = read_text_from_file(text_path, logger)
        embedding = dataToEmbedding(text, 'text', model, device)
        page_number = int(os.path.basename(text_path).split('_')[-1].split('.')[0])
        new_row = {
            'path': text_path,
            'media_type': 'text',
            'embeddings': embedding,
            'page_number': page_number
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        logger.debug(f"Embedding generado para el archivo de texto {text_path}.")

    logger.info("Generando embeddings para datos de imagen.")
    for image_path in tqdm(image_paths, desc="Procesando imágenes"):
        embedding = dataToEmbedding(image_path, 'image', model, device)
        new_row = {
            'path': image_path,
            'media_type': 'image',
            'embeddings': embedding,
            'page_number': None
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        logger.debug(f"Embedding generado para el archivo de imagen {image_path}.")

    logger.info("Embeddings generados para todos los datos.")
    return df

# Función para procesar el PDF e insertar embeddings
def process_pdf_and_insert_embeddings(pdf_content, model, device, database, table, logger):
    texts, image_paths = extract_pdf_contents(pdf_content, logger)
    df = create_embeddings_dataframe(texts, image_paths, model, device, logger)
    df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x, dtype=np.float32))
    df['page_number'] = df['page_number'].fillna(0).astype(np.int64)
    df['path'] = df['path'].astype(str)
    df['media_type'] = df['media_type'].astype(str)

    n = 200  
    logger.info("Insertando embeddings en KDB.ai.")
    for i in tqdm(range(0, df.shape[0], n), desc="Insertando en KDB.ai"):
        table.insert(df[i:i+n].reset_index(drop=True))
    logger.info("Todos los embeddings insertados en KDB.ai.")
    return df

# Función para convertir una consulta en embedding
def queryToEmbedding(text, model, device):
    txt = [text]
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(txt, device)
    }
    vec = getEmbeddingVector(inputs, model)
    return vec

# Función para preparar datos para RAG y obtener imágenes en base64
def RAG_Setup(results, retrieved_data_for_RAG, logger):
    images_base64 = []
    for index, row in results[0].iterrows():
        if row['media_type'] == 'image':
            try:
                image = Image.open(row['path'])
                retrieved_data_for_RAG.append(image)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images_base64.append(image_base64)
                logger.debug(f"Imagen {row['path']} añadida a los datos recuperados.")
            except Exception as e:
                logger.error(f"Error al abrir la imagen {row['path']}: {e}")
        elif row['media_type'] == 'text':
            text = read_text_from_file(row['path'], logger)
            if text is not None:
                retrieved_data_for_RAG.append(text)
                logger.debug(f"Texto {row['path']} añadido a los datos recuperados.")
    return retrieved_data_for_RAG, images_base64