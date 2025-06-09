from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import uuid
import fitz  # PyMuPDF
import boto3

app = FastAPI()
model = SentenceTransformer("clip-ViT-B-32")

# Configuración S3
S3_BUCKET = "rag-images-vida"
s3 = boto3.client("s3")

# Cliente de ChromaDB
chroma_client = HttpClient(host="localhost", port=8001)
collection = chroma_client.get_or_create_collection(name="multimodal")

# URL del LLM (Ngrok generado por el Colab)
LLM_API_URL = "https://4a25-34-168-246-220.ngrok-free.app/generate"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
def search_embeddings(request: QueryRequest):
    embedding = model.encode(request.query)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=request.top_k
    )

    best_image_url = None
    best_image_score = float("inf")

    for meta in results["metadatas"][0]:
        if "image_url" in meta:
            try:
                image = Image.open(requests.get(meta["image_url"], stream=True).raw).convert("RGB")
                image_embedding = model.encode(image)
                dist = np.linalg.norm(embedding - image_embedding)
                if dist < best_image_score:
                    best_image_score = dist
                    best_image_url = meta["image_url"]
            except Exception as e:
                print(f"Error con imagen sugerida: {e}")

    return {
        "results": [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ],
        "suggested_image": best_image_url
    }

@app.post("/query_llm")
def query_llm(request: QueryRequest):
    embedding = model.encode(request.query)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=request.top_k
    )

    context = "\n".join(results["documents"][0])

    # Buscar imagen sugerida
    best_image_url = None
    best_image_score = float("inf")
    for meta in results["metadatas"][0]:
        if "image_url" in meta:
            try:
                image = Image.open(requests.get(meta["image_url"], stream=True).raw).convert("RGB")
                image_embedding = model.encode(image)
                dist = np.linalg.norm(embedding - image_embedding)
                if dist < best_image_score:
                    best_image_score = dist
                    best_image_url = meta["image_url"]
            except Exception as e:
                print(f"Error con imagen sugerida: {e}")

    # Incluir markdown con imagen en el prompt
    imagen_md = f"\nImagen relacionada:\n![Imagen relacionada]({best_image_url})\n" if best_image_url else ""
    prompt = f"""Contesta la siguiente consulta con base en el contexto proporcionado. Puedes hacer referencia a la imagen si es útil.

Contexto:
{context}
{imagen_md}

Consulta:
{request.query}

Respuesta:"""

    try:
        headers = {
            "ngrok-skip-browser-warning": "true"
        }
        response = requests.post(
            LLM_API_URL,
            json={"prompt": prompt},
            headers=headers
        )
        response.raise_for_status()
        reply = response.json()["response"]
    except Exception as e:
        return {"error": f"No se pudo contactar al LLM: {e}"}

    return {
        "respuesta": reply,
        "suggested_image": best_image_url
    }

@app.post("/add")
async def add_document(
    text: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    pdf_file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(""),
    source_url: Optional[str] = Form(""),
    tags: Optional[str] = Form(""),
):
    content = ""
    image_embedding = None
    image_url = None

    if text:
        content = text

    elif image_file:
        img_bytes = await image_file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        content = f"Imagen cargada: {image_file.filename}"
        image_embedding = model.encode(img)

        uid = str(uuid.uuid4())
        s3_key = f"imagenes/{uid}_{image_file.filename}"
        s3.upload_fileobj(BytesIO(img_bytes), S3_BUCKET, s3_key)
        image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

    elif pdf_file:
        pdf_bytes = await pdf_file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        content = " ".join([page.get_text() for page in pdf])

    else:
        return {"error": "Debes enviar al menos un texto, imagen o PDF"}

    text_embedding = model.encode(content)
    final_embedding = ((text_embedding + image_embedding) / 2).tolist() if image_embedding else text_embedding.tolist()

    uid = str(uuid.uuid4())
    collection.add(
        documents=[content],
        ids=[uid],
        embeddings=[final_embedding],
        metadatas=[{
            "title": title,
            "source_url": source_url,
            "tags": tags,
            "image_url": image_url if image_url else None,
            "pdf_filename": pdf_file.filename if pdf_file else None
        }]
    )

    return {"message": "Documento añadido", "id": uid, "image_url": image_url}
