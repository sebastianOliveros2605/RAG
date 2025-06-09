import streamlit as st
import requests

SERVER_URL = "http://ec2-44-193-22-242.compute-1.amazonaws.com:8000"

st.set_page_config(page_title="Cliente RAG Médica", layout="wide")
st.title("🧠 Cliente RAG - Consulta médica con contexto visual")

tab1, tab2 = st.tabs(["🔍 Consultar", "➕ Agregar datos"])

with tab1:
    st.header("Buscar información médica")
    query = st.text_input("Ingresa tu pregunta o síntoma:", "")
    top_k = st.slider("Número de resultados", min_value=1, max_value=5, value=3)

    if st.button("Buscar"):
        if not query.strip():
            st.warning("Debes ingresar una consulta.")
        else:
            # 🔄 Usa el endpoint /query_llm (no /search)
            response = requests.post(
                f"{SERVER_URL}/query_llm",
                json={"query": query, "top_k": top_k}
            )

            if response.ok:
                data = response.json()
                st.subheader("Respuesta del modelo:")
                st.write(data.get("respuesta", "No se obtuvo respuesta."))
                print(data)
                # 👁 Mostrar imagen si viene
                image_url = data.get("suggested_image")
                if image_url:
                    st.markdown("### Imagen sugerida:")
                    st.image(image_url, use_column_width=True)
            else:
                st.error(f"Error al consultar: {response.text}")

with tab2:
    st.header("Agregar nuevo contenido")

    tipo = st.selectbox("Tipo de contenido", ["Texto", "Imagen", "PDF"])
    title = st.text_input("Título del documento (opcional):")
    source_url = st.text_input("Fuente o URL (opcional):")
    tags = st.text_input("Etiquetas (separadas por coma):")

    if tipo == "Texto":
        texto = st.text_area("Texto")
        if st.button("Agregar texto"):
            if texto.strip():
                resp = requests.post(
                    f"{SERVER_URL}/add",
                    data={
                        "text": texto,
                        "title": title,
                        "source_url": source_url,
                        "tags": tags,
                    }
                )
                st.success(resp.json())
            else:
                st.warning("El texto no puede estar vacío.")

    elif tipo == "Imagen":
        imagen = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
        if imagen and st.button("Agregar imagen"):
            files = {"image_file": imagen}
            data = {"title": title, "source_url": source_url, "tags": tags}
            resp = requests.post(f"{SERVER_URL}/add", files=files, data=data)
            st.success(resp.json())

    elif tipo == "PDF":
        pdf = st.file_uploader("Sube un archivo PDF", type=["pdf"])
        if pdf and st.button("Agregar PDF"):
            files = {"pdf_file": pdf}
            data = {"title": title, "source_url": source_url, "tags": tags}
            resp = requests.post(f"{SERVER_URL}/add", files=files, data=data)
            st.success(resp.json())
