import streamlit as st
import requests
import re

SERVER_URL = "http://ec2-44-193-22-242.compute-1.amazonaws.com:8000"

# 🧠 Configuración general
st.set_page_config(page_title="Asistente con RAG", layout="centered")
st.markdown("""
    <style>
        .chat-response {
            background-color: #f1f3f5;
            padding: 1.2rem;
            border-radius: 12px;
            font-size: 1.1rem;
            margin-top: 1rem;
        }
        .section-title {
            font-weight: bold;
            font-size: 1.2rem;
            margin-top: 1.5rem;
        }
        .user-input {
            font-size: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🏥 Título principal
st.title("🧠 Asistente con RAG")

# 🔍 Entrada del usuario
query = st.text_input("💬 ¿Cuál es tu consulta?", "", placeholder="Ej: ¿Qué es Machu Pichu?")
top_k = st.slider("🔎 ¿Cuántas fuentes quieres consultar?", min_value=1, max_value=5, value=3)

col_button = st.container()
with col_button:
    if st.button("🚀 Obtener Respuesta", use_container_width=True, key="search_button_main"):
        if not query.strip():
            st.warning("¡Oops! Parece que olvidaste escribir tu pregunta. Intenta de nuevo.")
        else:
            with st.spinner("Buscando y tejiendo una respuesta inteligente..."):
                try:
                    response = requests.post(
                        f"{SERVER_URL}/query_llm",
                        json={"query": query, "top_k": top_k},
                        timeout=360
                    )
                    response.raise_for_status()

                    data = response.json()

                    # --- Expander con detalles técnicos ---
                    with st.expander("🧪 Ver Detalles Técnicos (para desarrolladores)"):
                        st.json(data)
                        st.write(f"URL del servidor: {SERVER_URL}")

                    # --- Procesamiento de respuesta ---
                    raw_response = data.get("respuesta", "")
                    clean_response = ""

                    match = re.search(r"model (.*)", raw_response, re.DOTALL)
                    if match:
                        clean_response = match.group(1).strip()
                    else:
                        clean_response = raw_response.strip()

                    if clean_response:
                        st.markdown('<p class="section-title">🤖 Respuesta del asistente:</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-response">{clean_response}</div>', unsafe_allow_html=True)
                    else:
                        st.info("Lo siento, no pude encontrar una respuesta relevante. Intenta reformular tu pregunta.")

                    # --- Imagen sugerida (si aplica) ---
                    image_url = data.get("suggested_image")
                    if image_url:
                        if not image_url.startswith("http"):
                            if image_url.startswith("/"):
                                image_url = f"{SERVER_URL}{image_url}"
                            else:
                                image_url = f"{SERVER_URL}/images/{image_url}"

                        st.markdown('<p class="section-title">🖼️ Imagen relacionada:</p>', unsafe_allow_html=True)
                        st.image(image_url.strip(), use_container_width=True, caption="Sugerida por el modelo")
                    else:
                        st.info("No se encontró una imagen relevante para esta consulta.")

                except requests.exceptions.RequestException as e:
                    st.error("⚠️ ¡Error al contactar el servidor! Asegúrate de que esté disponible.")
                    if 'response' in locals() and response is not None:
                        with st.expander("📋 Información adicional del error"):
                            st.write(f"Código HTTP: {response.status_code}")
                            st.write(f"Respuesta del servidor: {response.text}")
                    st.exception(e)
                except Exception as e:
                    st.error("❌ Ha ocurrido un error inesperado.")
                    st.exception(e)

# 🔚 Pie de página
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🧠 Impulsado por RAG Multimodal • 2025</p>", unsafe_allow_html=True)
