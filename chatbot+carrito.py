#!/usr/bin/env python3
"""
Chatbot de búsqueda de CVs con Streamlit
Interfaz de usuario para el agente especialista en currículums

CONFIGURACIÓN:
- Modifica las variables en initialize_session_state() para tu entorno
- Asegúrate de tener configurada la variable OPENAI_API_KEY en tu entorno
"""

import streamlit as st
import os
from datetime import datetime
import json
from typing import Dict, List, Any
import sys

# Importar el agente de búsqueda
from cv_search_agent import CVSearchAgentWithMemory, SearchConfig


# Configuración de la página
st.set_page_config(
    page_title="🤖 Asistente de Reclutamiento",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .stChat {
        background-color: #f0f2f5;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #007bff;
        color: white;
    }
    .chat-message.assistant {
        background-color: white;
        border: 1px solid #e0e0e0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    .candidate-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa las variables de estado de la sesión"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agent' not in st.session_state:
        # Crear el agente automáticamente con la configuración hardcodeada
        config = SearchConfig(
            db_uri="./data/target_db",  # Configurar aquí tu BD
            table_name="personal_embeddings",  # Configurar aquí tu tabla
            openai_api_key=os.getenv("OPENAI_API_KEY", "sk-..."),  # Configurar aquí tu API key o usar variable de entorno
            llm_model="gpt-4",  # o "gpt-3.5-turbo" para menor costo
            temperature=0.3,
            top_k=5
        )
        
        try:
            st.session_state.agent = CVSearchAgentWithMemory(
                config=config,
                memory_size=10
            )
        except Exception as e:
            st.error(f"Error al inicializar el agente: {str(e)}")
            st.stop()
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'total_searches' not in st.session_state:
        st.session_state.total_searches = 0
    
    if 'shopping_cart' not in st.session_state:
        st.session_state.shopping_cart = []
    
    if 'show_candidate_details' not in st.session_state:
        st.session_state.show_candidate_details = None


def create_agent(config: SearchConfig):
    """Crea o actualiza el agente de búsqueda"""
    try:
        st.session_state.agent = CVSearchAgentWithMemory(
            config=config,
            memory_size=10
        )
        return True
    except Exception as e:
        st.error(f"Error al crear el agente: {str(e)}")
        return False


def display_message(role: str, content: str, timestamp: str = None):
    """Muestra un mensaje en el chat"""
    avatar = "👤" if role == "user" else "🤖"
    name = "Tú" if role == "user" else "Asistente"
    
    with st.chat_message(role, avatar=avatar):
        if timestamp:
            st.caption(f"_{timestamp}_")
        st.markdown(content)


def display_candidates(candidates: List[Dict[str, Any]]):
    """Muestra los candidatos encontrados en cards con botones interactivos"""
    if not candidates:
        return
    
    st.markdown("### 📋 Candidatos Encontrados")
    
    # Crear columnas para las tarjetas
    cols = st.columns(min(len(candidates), 3))
    
    for idx, candidate in enumerate(candidates[:6]):  # Máximo 6 candidatos
        col = cols[idx % 3]
        
        with col:
            # Crear un contenedor para cada tarjeta
            with st.container():
                # Card con información del candidato
                st.markdown(f"""
                <div class="candidate-card">
                    <h4>{candidate.get('name', 'N/A')}</h4>
                    <p><strong>📧</strong> {candidate.get('email', 'N/A')}</p>
                    <p><strong>💼</strong> {candidate.get('position', 'N/A')}</p>
                    <p><strong>🎯</strong> Coincidencia: {candidate.get('score', 0)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Crear un ID único para el candidato
                candidate_id = f"{candidate.get('email', '')}_{idx}"
                
                # Contenedor para los botones
                button_cols = st.columns(2)
                
                with button_cols[0]:
                    # Botón para agregar al carrito
                    if st.button(
                        "🛒 Agregar",
                        key=f"add_{candidate_id}",
                        help="Agregar candidato a la lista de seleccionados",
                        use_container_width=True
                    ):
                        add_to_cart(candidate)
                
                with button_cols[1]:
                    # Botón para ver más información
                    if st.button(
                        "ℹ️ Más info",
                        key=f"info_{candidate_id}",
                        help="Ver información detallada del candidato",
                        use_container_width=True
                    ):
                        show_candidate_details(candidate)
                
                st.markdown("---")


def add_to_cart(candidate: Dict[str, Any]):
    """Agrega un candidato al carrito de selección"""
    # Verificar si ya está en el carrito
    if not any(c.get('email') == candidate.get('email') for c in st.session_state.shopping_cart):
        st.session_state.shopping_cart.append(candidate)
        st.success(f"✅ {candidate.get('name', 'Candidato')} agregado a la selección")
    else:
        st.warning(f"⚠️ {candidate.get('name', 'Candidato')} ya está en la selección")


def show_candidate_details(candidate: Dict[str, Any]):
    """Muestra información detallada del candidato en un modal"""
    st.session_state.show_candidate_details = candidate


def display_candidate_modal():
    """Muestra un modal con información detallada del candidato"""
    if st.session_state.show_candidate_details:
        candidate = st.session_state.show_candidate_details
        
        # Crear un expander que actúa como modal
        with st.expander(f"📋 Información Detallada - {candidate.get('name', 'N/A')}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {candidate.get('name', 'N/A')}")
                st.markdown(f"**Cargo:** {candidate.get('position', 'N/A')}")
                st.markdown(f"**Email:** {candidate.get('email', 'N/A')}")
                st.markdown(f"**Score de Coincidencia:** {candidate.get('score', 0)*100:.1f}%")
                
                if candidate.get('summary'):
                    st.markdown("#### 📝 Resumen Profesional")
                    st.text_area("", value=candidate.get('summary', ''), height=150, disabled=True)
                
                # Solicitar más información al agente
                if st.button("🤖 Solicitar análisis detallado"):
                    query = f"Dame más información detallada sobre el candidato {candidate.get('name', '')} con email {candidate.get('email', '')}"
                    # Simular envío de mensaje
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.session_state.show_candidate_details = None
                    st.rerun()
            
            with col2:
                st.markdown("### Acciones")
                if st.button("🛒 Agregar a selección", use_container_width=True):
                    add_to_cart(candidate)
                
                if st.button("❌ Cerrar", use_container_width=True):
                    st.session_state.show_candidate_details = None
                    st.rerun()


def sidebar_configuration():
    """Configuración en el sidebar"""
    with st.sidebar:
        st.title("🤖 Asistente de Reclutamiento")
        
        # Carrito de compras / Candidatos seleccionados
        st.divider()
        st.subheader(f"🛒 Candidatos Seleccionados ({len(st.session_state.shopping_cart)})")
        
        if st.session_state.shopping_cart:
            for i, candidate in enumerate(st.session_state.shopping_cart):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"• {candidate.get('name', 'N/A')}")
                        st.caption(f"  {candidate.get('position', '')}")
                    with col2:
                        if st.button("❌", key=f"remove_{i}", help="Quitar de la selección"):
                            st.session_state.shopping_cart.pop(i)
                            st.rerun()
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Exportar Lista", use_container_width=True):
                    export_selected_candidates()
            with col2:
                if st.button("🗑️ Limpiar Todo", use_container_width=True):
                    st.session_state.shopping_cart = []
                    st.rerun()
        else:
            st.info("No hay candidatos seleccionados")
        
        # Estadísticas
        st.divider()
        st.subheader("📊 Estadísticas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Mensajes", len(st.session_state.messages))
        with col2:
            st.metric("🔍 Búsquedas", st.session_state.total_searches)
        
        # Historial de búsquedas
        if st.session_state.search_history:
            st.divider()
            st.subheader("📜 Historial Reciente")
            for search in st.session_state.search_history[-5:]:
                st.text(f"• {search[:30]}...")
        
        # Botón para limpiar chat
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Limpiar Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.agent:
                    st.session_state.agent.clear_memory()
                st.rerun()
        
        with col2:
            if st.button("🔄 Reiniciar", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Información
        st.divider()
        with st.expander("ℹ️ Cómo usar"):
            st.markdown("""
            **Ejemplos de búsquedas:**
            
            • "Necesito un desarrollador Python con experiencia en IA"
            • "Busco un Scrum Master certificado"
            • "Consultor SAP con experiencia en minería"
            • "Technical Lead con conocimientos en cloud"
            
            **Tips:**
            - Sé específico en tus requisitos
            - Menciona las habilidades clave
            - Indica la experiencia deseada
            - Especifica industrias si es relevante
            
            **Nuevas funciones:**
            - 🛒 Agregar candidatos a tu selección
            - ℹ️ Ver información detallada de cada perfil
            - 📥 Exportar lista de seleccionados
            
            **Preguntas de seguimiento:**
            - "Dame más detalles del segundo candidato"
            - "¿Cuál tiene más experiencia en cloud?"
            - "Compara los primeros dos candidatos"
            - "¿Alguno ha trabajado en startups?"
            """)
        
        # About
        st.divider()
        with st.expander("📋 Acerca de"):
            st.markdown("""
            **Asistente de Reclutamiento v1.0**
            
            Sistema inteligente de búsqueda de talento
            con IA y búsqueda semántica.
            
            - 🧠 Powered by OpenAI GPT-4
            - 🔍 Búsqueda vectorial con LanceDB
            - 💬 Memoria de conversación
            - 📊 Análisis experto de perfiles
            - 🛒 Gestión de candidatos seleccionados
            """)
        
        # Footer con información del sistema
        st.divider()
        st.caption("🟢 Sistema Activo")
        st.caption(f"Modelo: GPT-4")
        st.caption(f"Memoria: 10 mensajes")


def export_selected_candidates():
    """Exporta la lista de candidatos seleccionados"""
    if st.session_state.shopping_cart:
        import pandas as pd
        
        # Crear DataFrame con los candidatos
        df_data = []
        for candidate in st.session_state.shopping_cart:
            df_data.append({
                'Nombre': candidate.get('name', ''),
                'Email': candidate.get('email', ''),
                'Cargo': candidate.get('position', ''),
                'Score': f"{candidate.get('score', 0)*100:.1f}%",
                'Resumen': candidate.get('summary', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Crear CSV para descarga
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"candidatos_seleccionados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Lista preparada para descarga")

def main():
    """Función principal de la aplicación"""
    
    # Inicializar estado de la sesión
    initialize_session_state()
    
    # Configuración del sidebar
    sidebar_configuration()
    
    # Header principal
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🤖 Asistente de Reclutamiento")
        st.markdown("*Especialista en búsqueda inteligente de talento*")
    
    st.divider()
    
    # Contenedor principal del chat
    chat_container = st.container()
    
    # Mostrar modal de detalles si está activo
    display_candidate_modal()
    
    # Mostrar historial de mensajes
    with chat_container:
        for message in st.session_state.messages:
            display_message(
                message["role"],
                message["content"],
                message.get("timestamp")
            )
            
            # Si el mensaje tiene candidatos, mostrarlos
            if "candidates" in message:
                display_candidates(message["candidates"])
    
    # Input del usuario
    if prompt := st.chat_input("¿Qué tipo de profesional estás buscando?"):
        # Añadir mensaje del usuario
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Mostrar mensaje del usuario
        display_message("user", prompt, timestamp)
        
        # Añadir a historial de búsquedas
        st.session_state.search_history.append(prompt)
        st.session_state.total_searches += 1
        
        # Mostrar spinner mientras se procesa
        with st.spinner("🔍 Buscando candidatos..."):
            try:
                # Realizar búsqueda con el agente
                response = st.session_state.agent.search_with_context(prompt)
                
                # Preparar respuesta del asistente
                assistant_message = {
                    "role": "assistant",
                    "content": response.get('analysis', 'No se pudo procesar la búsqueda.'),
                    "timestamp": datetime.now().strftime("%H:%M")
                }
                
                # Si hay candidatos, añadirlos al mensaje
                if response.get('candidates'):
                    assistant_message["candidates"] = response['candidates']
                
                # Añadir respuesta del asistente
                st.session_state.messages.append(assistant_message)
                
                # Mostrar respuesta del asistente
                display_message("assistant", assistant_message["content"], assistant_message["timestamp"])
                
                # Mostrar candidatos si los hay
                if response.get('candidates'):
                    display_candidates(response['candidates'])
                
                # Mostrar métricas de la búsqueda
                if response.get('total_candidates', 0) > 0:
                    if response.get('is_followup'):
                        st.info(f"💬 Respondiendo sobre los {response['total_candidates']} candidatos anteriores")
                    else:
                        st.success(f"✅ Se encontraron {response['total_candidates']} candidatos relevantes")
                
            except Exception as e:
                error_message = f"❌ Error al procesar la búsqueda: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.error(error_message)
    
    # Footer con información adicional
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("💡 Tip: Puedes hacer preguntas de seguimiento sobre los candidatos mostrados")


if __name__ == "__main__":
    main()