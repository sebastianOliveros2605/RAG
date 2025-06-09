import wikipedia
import requests
import uuid
import boto3
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient
import os
import re

# --- Configuración S3 ---
BUCKET_NAME = "rag-images-vida"
s3 = boto3.client("s3")

current_directory = os.getcwd()
print("======================================================")
print(f"📣 Directorio de trabajo actual: {current_directory}")
print("======================================================")

# --- Conexión a ChromaDB en modo servidor ---
chroma_client = HttpClient(host="localhost", port=8081)
collection = chroma_client.get_or_create_collection(name="multimodal")

# --- Modelo multimodal ---
model = SentenceTransformer("clip-ViT-B-32")

# --- Idioma de Wikipedia ---
wikipedia.set_lang("es")

# --- Temas a poblar ---
topics = [
    "Machu Picchu",
    "Cultura de Japón",
    "Desierto del Sáhara",
    "Selva amazónica",
    "Civilización maya",
    "Cordillera de los Andes",
    "Antiguo Egipto",
    "Monte Fuji",
    "Antártida",
    "Colombia",
    "Cultura de Europa",
    "Estados Unidos",
    "Continente",
    "Organización de las Naciones Unidas",
    "África",
    "Nuevas siete maravillas del mundo moderno"
]

# --- User-Agent obligatorio para Wikipedia ---
headers = {
    'User-Agent': 'MyCoolRAGProject/1.0 (johanoliveros99@gmail.com; +https://mi-proyecto.com) python-requests/2.26.0'
}

# --- Funciones auxiliares ---
def limpiar_texto(texto):
    texto = re.sub(r"\[[^\]]*\]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[\u200b\u200e\u200f]", "", texto)
    return texto.strip()

def dividir_en_chunks(texto, tamaño_chunk=400, solapamiento=100):
    chunks = []
    i = 0
    while i < len(texto):
        fin = min(i + tamaño_chunk, len(texto))
        chunks.append(texto[i:fin])
        i += tamaño_chunk - solapamiento
    return chunks

def upload_to_s3(img_url):
    try:
        response = requests.get(img_url, headers=headers)
        response.raise_for_status()
        
        # Validar que sea imagen
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img.verify()  # Esto lanza excepción si no es una imagen válida
        
        # Si pasa, subimos a S3
        img_data.seek(0)
        file_name = f"{uuid.uuid4()}.jpg"
        s3.upload_fileobj(img_data, BUCKET_NAME, file_name)
        return f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        print(f"❌ Error al verificar o subir imagen a S3: {e}")
        return None


def get_valid_image_url(images):
    for img_url in images:
        if img_url.lower().endswith((".jpg", ".jpeg", ".png")) and "wikimedia" in img_url:
            return img_url
    return None

# --- Proceso principal ---
for topic in topics:
    try:
        page = wikipedia.page(topic, auto_suggest=True, redirect=True)
        summary = wikipedia.summary(page.title, sentences=50)
        texto_limpio = limpiar_texto(summary)
        chunks = dividir_en_chunks(texto_limpio)

        image_url = get_valid_image_url(page.images)
        if not image_url:
            print(f"⚠️ No se encontró imagen válida para: {topic}")
            continue

        s3_url = upload_to_s3(image_url)
        if not s3_url:
            print(f"⚠️ No se pudo subir imagen para: {topic}")
            continue

        response = requests.get(image_url, headers=headers)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        embedding_image = model.encode(image)

        for chunk in chunks:
            embedding_text = model.encode(chunk)
            final_embedding = ((embedding_text + embedding_image) / 2).tolist()

            collection.add(
                documents=[chunk],
                ids=[str(uuid.uuid4())],
                metadatas=[{
                    "topic": topic,
                    "wikipedia_url": page.url,
                    "image_url": s3_url
                }],
                embeddings=[final_embedding]
            )
        print(f"✅ Insertado: {topic} ({len(chunks)} chunks)")

    except wikipedia.exceptions.PageError:
        print(f"❌ Error con {topic}: La página no existe.")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"❌ Error con {topic}: Desambiguación. Opciones: {e.options[:5]}")
    except Exception as e:
        print(f"❌ Error inesperado con {topic}: {e}")

print("🔍 Llegamos al final del for")
