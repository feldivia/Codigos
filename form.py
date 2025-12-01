# Front-end Formulario propuestas con carga de documentos
from genia.ppt_generador import *
from genia.prompt_template import *
from lib.config import *
from lib.var_globales import *

import streamlit as st
from datetime import date
from lib.listas_formularios import SUBLOS, SERVICIOS_POR_SUBLOS, INDUSTRIAS, PERSONAS_POR_SUBLOS, DIFERENCIADORES
import pandas as pd
import base64
import os
import tempfile
from typing import List, Dict, Optional

# Importar el módulo de procesamiento de documentos
from document_processor import (
    DocumentProcessor, 
    MultiDocumentProcessor,
    ProcessingResult,
    ProcessingStatus,
    ExtractedContent,
    save_uploaded_file,
    cleanup_temp_files,
    get_file_icon,
    format_keywords,
    check_dependencies
)


def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def render_processing_status(results: List[ProcessingResult]) -> None:
    """Renderiza el estado de procesamiento de cada documento con semáforos"""
    st.markdown("##### 📋 Estado de Procesamiento")
    
    for result in results:
        icon = get_file_icon(result.filename)
        
        if result.status == ProcessingStatus.SUCCESS:
            # Semáforo verde
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    padding: 8px 12px;
                    margin: 5px 0;
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                    border-radius: 4px;
                ">
                    <span style="
                        width: 12px;
                        height: 12px;
                        background-color: #28a745;
                        border-radius: 50%;
                        margin-right: 10px;
                        box-shadow: 0 0 5px #28a745;
                    "></span>
                    <span style="font-size: 0.9rem;">{icon} {result.filename}</span>
                    <span style="margin-left: auto; color: #155724; font-size: 0.8rem;">✓ Procesado</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Semáforo rojo
            error_msg = result.error_message[:50] + "..." if len(result.error_message) > 50 else result.error_message
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    padding: 8px 12px;
                    margin: 5px 0;
                    background-color: #f8d7da;
                    border-left: 4px solid #dc3545;
                    border-radius: 4px;
                ">
                    <span style="
                        width: 12px;
                        height: 12px;
                        background-color: #dc3545;
                        border-radius: 50%;
                        margin-right: 10px;
                        box-shadow: 0 0 5px #dc3545;
                    "></span>
                    <span style="font-size: 0.9rem;">{icon} {result.filename}</span>
                    <span style="margin-left: auto; color: #721c24; font-size: 0.8rem;">✗ Error</span>
                </div>
                <div style="
                    padding: 5px 12px 5px 34px;
                    font-size: 0.75rem;
                    color: #721c24;
                ">{error_msg}</div>
                """,
                unsafe_allow_html=True
            )


def render_document_sidebar(logo_base64: str) -> Optional[ExtractedContent]:
    """
    Renderiza la columna lateral para carga de documentos.
    
    Returns:
        ExtractedContent consolidado si se procesaron documentos, None en caso contrario
    """
    st.markdown(
        """
        <style>
        .sidebar-header {
            background: linear-gradient(135deg, #FE7C39 0%, #e55a1c 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        .sidebar-header h3 {
            margin: 0;
            font-size: 1.1rem;
        }
        .upload-zone {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            background-color: #f9f9f9;
            margin: 10px 0;
        }
        .upload-zone:hover {
            border-color: #FE7C39;
            background-color: #fff5f0;
        }
        .file-count-badge {
            background-color: #FE7C39;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-left: 5px;
        }
        .processing-info {
            background-color: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header del sidebar
    st.markdown(
        """
        <div class="sidebar-header">
            <h3>📁 Carga de Documentos</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Información sobre tipos de archivo
    st.markdown(
        """
        <div class="processing-info">
            <strong>Formatos soportados:</strong><br>
            📄 PDF &nbsp;&nbsp; 📝 Word (.docx) &nbsp;&nbsp; 📊 PowerPoint (.pptx)
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader (máximo 5 archivos)
    uploaded_files = st.file_uploader(
        "Arrastra o selecciona documentos",
        type=['pdf', 'docx', 'pptx'],
        accept_multiple_files=True,
        key="document_uploader",
        help="Puedes cargar hasta 5 documentos de tipo PDF, Word o PowerPoint"
    )
    
    # Validar cantidad de archivos
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠️ Máximo 5 documentos. Se procesarán solo los primeros 5.")
        uploaded_files = uploaded_files[:5]
    
    # Mostrar archivos cargados
    if uploaded_files:
        st.markdown(f"**Archivos cargados:** <span class='file-count-badge'>{len(uploaded_files)}</span>", unsafe_allow_html=True)
        
        for file in uploaded_files:
            icon = get_file_icon(file.name)
            size_kb = file.size / 1024
            st.markdown(f"&nbsp;&nbsp;{icon} {file.name} ({size_kb:.1f} KB)")
    
    st.divider()
    
    # Botón de procesar
    process_button = st.button(
        "🔍 Procesar Documentos",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
        key="btn_process_docs"
    )
    
    # Estado de procesamiento
    if "doc_processing_results" not in st.session_state:
        st.session_state.doc_processing_results = None
    if "doc_consolidated_content" not in st.session_state:
        st.session_state.doc_consolidated_content = None
    
    # Procesar documentos cuando se presiona el botón
    if process_button and uploaded_files:
        with st.spinner("Procesando documentos..."):
            # Crear carpeta temporal
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            try:
                # Guardar archivos temporalmente
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file, temp_dir)
                    file_paths.append(file_path)
                
                # Procesar documentos
                multi_processor = MultiDocumentProcessor()
                results = multi_processor.process_documents(file_paths)
                
                # Guardar resultados en session_state
                st.session_state.doc_processing_results = results
                st.session_state.doc_consolidated_content = multi_processor.consolidate_content()
                
            except Exception as e:
                st.error(f"Error durante el procesamiento: {str(e)}")
            finally:
                # Limpiar archivos temporales
                cleanup_temp_files(temp_dir)
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
    
    # Mostrar resultados de procesamiento
    if st.session_state.doc_processing_results:
        st.divider()
        render_processing_status(st.session_state.doc_processing_results)
        
        # Mostrar resumen
        success_count = sum(1 for r in st.session_state.doc_processing_results if r.status == ProcessingStatus.SUCCESS)
        total_count = len(st.session_state.doc_processing_results)
        
        if success_count > 0:
            st.success(f"✅ {success_count}/{total_count} documentos procesados correctamente")
            
            # Checkbox para autocompletar automáticamente
            auto_apply = st.checkbox(
                "✨ Autocompletar formulario automáticamente",
                value=True,
                key="chk_auto_apply",
                help="Si está activado, la información extraída se aplicará automáticamente al formulario"
            )
            
            # Si el autocompletado está activado y hay contenido nuevo, aplicar automáticamente
            if auto_apply and st.session_state.doc_consolidated_content:
                if not st.session_state.get("content_already_applied", False):
                    st.session_state.apply_extracted_content = True
                    st.session_state.content_already_applied = True
                    st.rerun()
            
            # Botón manual para re-aplicar si es necesario
            if st.button("🔄 Re-aplicar al Formulario", type="secondary", use_container_width=True, key="btn_apply_content"):
                st.session_state.apply_extracted_content = True
                st.session_state.content_already_applied = True
                st.rerun()
            
            # Mostrar preview de la información extraída
            with st.expander("👁️ Ver información extraída", expanded=False):
                content = st.session_state.doc_consolidated_content
                if content:
                    if content.titulo_propuesta:
                        st.markdown(f"**Título:** {content.titulo_propuesta[:100]}...")
                    if content.cliente:
                        st.markdown(f"**Cliente:** {content.cliente}")
                    if content.industria:
                        st.markdown(f"**Industria:** {content.industria}")
                    if content.palabras_claves:
                        st.markdown(f"**Keywords:** {', '.join(content.palabras_claves[:5])}...")
                    if content.problema:
                        st.markdown(f"**Problema:** {content.problema[:150]}...")
                    if content.objetivo_general:
                        st.markdown(f"**Objetivo:** {content.objetivo_general[:150]}...")
        
        if success_count < total_count:
            st.warning(f"⚠️ {total_count - success_count} documento(s) con errores")
    
    # Botón para limpiar
    if st.session_state.doc_processing_results:
        st.divider()
        if st.button("🗑️ Limpiar Resultados", use_container_width=True, key="btn_clear_results"):
            st.session_state.doc_processing_results = None
            st.session_state.doc_consolidated_content = None
            st.session_state.content_already_applied = False
            st.rerun()
    
    return st.session_state.doc_consolidated_content


def apply_extracted_content_to_form(content: ExtractedContent) -> None:
    """Aplica el contenido extraído a los campos del formulario"""
    if not content:
        return
    
    # Solo aplicar si hay valores no vacíos
    if content.titulo_propuesta:
        st.session_state.titulo_propuesta = content.titulo_propuesta
    
    if content.palabras_claves:
        st.session_state.palabras_claves = format_keywords(content.palabras_claves)
    
    if content.cliente:
        st.session_state.cliente = content.cliente
    
    if content.industria:
        # Buscar la industria en la lista de industrias disponibles
        industria_lower = content.industria.lower()
        for ind in INDUSTRIAS:
            if industria_lower in ind.lower() or ind.lower() in industria_lower:
                st.session_state.industria = ind
                break
    
    if content.problema:
        st.session_state.problema = content.problema
    
    if content.objetivo_general:
        st.session_state.objetivo_general = content.objetivo_general
    
    if content.objetivos_secundarios:
        st.session_state.objetivo_secundario = ". ".join(content.objetivos_secundarios)
    
    if content.alcance_funcional:
        st.session_state.alcance_funcional = content.alcance_funcional
    
    if content.alcance_tecnico:
        st.session_state.alcance_tecnico = content.alcance_tecnico
    
    if content.alcance_geografico:
        st.session_state.alcance_geo_org = content.alcance_geografico
    
    if content.limitaciones:
        st.session_state.alcance_limitaciones = content.limitaciones


def pantalla_generacion_propuesta(input_data, logo_base64):
    """Pantalla que reemplaza al formulario mientras se genera la propuesta"""
   
    # =========================
    # ESTILOS CSS (MISMOS DEL FORMULARIO + CONTENEDOR FIJO)
    # =========================
    st.markdown(
        f"""
        <style>

        .logo-container {{
            position: absolute;
            top: 30px;
            right: 0px;
            z-index: 100;
        }}
        header, [data-testid="stHeader"] {{
            display: none !important;
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            height: 100%;
            width:100%;
            background-color: #DFE3E6 !important;
        }}
        .block-container {{
            max-width: 95%;
            padding: 2% 0.5%;
            background-color: transparent;
            padding-bottom: 150px !important; /* Espacio para el contenedor fijo */
        }}
       
        /* Contenedor fijo inferior para estado de generación */
        .status-container {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;

            background-color: #E8E8E8;
            border-top: 3px solid #FE7C39;
            padding: 20px 30px;
            z-index: 999;
            box-shadow: 0 -4px 12px rgba(0,0,0,0.15);
        }}
       
        .status-title {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }}
       
        .status-message {{
            font-size: 1rem;
            color: #555;
            display: flex;
            align-items: center;
        }}
       
        .status-icon {{
            margin-right: 10px;
            font-size: 1.3rem;
        }}
       
        .status-spinner {{

            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #FE7C39;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }}
       
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )
 
    # Banner superior (MISMO DEL FORMULARIO)
    st.markdown(
        """
        <div style="
            position: fixed;

            top: 0;
            left: 0;
            right: 0;          
            background-color: #FE7C39;
            color: white;
            padding: 0rem 2rem;
            z-index: 1000;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h1 style="margin: 0; font-size: 1.5rem;">Propuestas Automatizadas con GenAI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    # Espacio para que el contenido no quede detrás del banner
    st.markdown("<br>", unsafe_allow_html=True)
   
    # =========================
    # CONTENEDOR PARA ESTADO DE PROCESOS
    # =========================
    status_placeholder = st.empty()
   
    def actualizar_estado(estado, mensaje):

        """
        Actualiza el contenedor de estado
        estado: 'cargando', 'generando', 'terminado', 'error'
        mensaje: texto a mostrar
        """
        if estado == 'cargando':
            icon = "📦"
            status_html = f"""
            <div class="status-container">
                <div class="status-title">Estado del Proceso</div>
                <div class="status-message">
                    <span class="status-icon">{icon}</span>
                    <span>{mensaje}</span>
                </div>
            </div>
            """
        elif estado == 'generando':
            status_html = f"""
            <div class="status-container">
                <div class="status-title">Estado del Proceso</div>
                <div class="status-message">
                    <div class="status-spinner"></div>
                    <span style="color: #FE7C39; font-weight: bold;">{mensaje}</span>
                </div>
            </div>
            """
        elif estado == 'terminado':

            icon = "✅"
            status_html = f"""
            <div class="status-container" style="background-color: #d4edda; border-top: 3px solid #28a745;">
                <div class="status-title" style="color: #155724;">Estado del Proceso</div>
                <div class="status-message">                    
                    <span style="color: #155724; font-weight: bold;">{mensaje}</span>
                </div>
            </div>
            """
        elif estado == 'error':
            icon = "❌"
            status_html = f"""
            <div class="status-container" style="background-color: #f8d7da; border-top: 3px solid #dc3545;">
                <div class="status-title" style="color: #721c24;">Estado del Proceso</div>
                <div class="status-message">
                    <span class="status-icon">{icon}</span>
                    <span style="color: #721c24; font-weight: bold;">{mensaje}</span>
                </div>
            </div>
            """
       
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
   
    # =========================

    # ESTADO 1: CARGANDO PÁGINA
    # =========================
    if "estado_generacion" not in st.session_state:
        st.session_state.estado_generacion = "cargando"
        nombre_propuesta = input_data.get('titulo_propuesta', 'Propuesta')
        st.session_state.nombre_propuesta = nombre_propuesta
        actualizar_estado('cargando', f"Preparando generación de: '{nombre_propuesta}'")
   
    # =========================
    # CONTENIDO DE GENERACIÓN
    # =========================
    st.markdown("### ⏳ Generando Propuesta")
    st.markdown("Se está generando su propuesta. Este proceso puede tomar entre 2-5 minutos.")
    st.markdown(f"**Propuesta:** {st.session_state.nombre_propuesta}")
    st.markdown(f"**Cliente:** {input_data.get('cliente', 'N/A')}")
    st.markdown(f"**Solicitado por:** {input_data.get('solicitado_por', 'N/A')}")
   
    # =========================
    # ESTADO 2: GENERANDO PROPUESTA
    # =========================
    if "generacion_en_ejecucion" not in st.session_state:
        st.session_state.generacion_en_ejecucion = True
        st.session_state.estado_generacion = "generando"
       
        # Actualizar estado a "generando"

        actualizar_estado('generando', f"⏳ Generando propuesta '{st.session_state.nombre_propuesta}'... Por favor espere.")
       
        # Conexión GenAI
        if ambiente == 'LOCAL':
            obj_llm = GenAIConnection(get_secrets("OPENAI_API_KEY"), get_secrets("OPENAI_API_BASE"))
        elif ambiente == 'GCAAS':
            obj_llm = GenAIConnection(OPENAI_API_KEY, OPENAI_API_BASE)
       
        try:
            # Llamar a la función de generación
            generar_propuesta_ppt(obj_llm, input_data)
           
            # =========================
            # ESTADO 3: PROCESO TERMINADO
            # =========================
            st.session_state.estado_generacion = "terminado"
            st.session_state.generacion_finalizada = True
            actualizar_estado('terminado', f"✅ Propuesta '{st.session_state.nombre_propuesta}' generada exitosamente")
           
        except Exception as e:
            st.session_state.estado_generacion = "error"
            st.session_state.generacion_error = str(e)
            actualizar_estado('error', f"❌ Error al generar la propuesta: {str(e)}")
   
    else:

        # Mantener el estado actual en pantalla
        if st.session_state.estado_generacion == "cargando":
            actualizar_estado('cargando', f"Preparando generación de: '{st.session_state.nombre_propuesta}'")
        elif st.session_state.estado_generacion == "generando":
            actualizar_estado('generando', f"⏳ Generando propuesta '{st.session_state.nombre_propuesta}'... Por favor espere.")
        elif st.session_state.estado_generacion == "terminado":
            actualizar_estado('terminado', f"✅ Propuesta '{st.session_state.nombre_propuesta}' generada exitosamente")
        elif st.session_state.estado_generacion == "error":
            actualizar_estado('error', f"❌ Error al generar la propuesta: {st.session_state.get('generacion_error', 'Error desconocido')}")
   
    # =========================
    # MENSAJES POST-PROCESO
    # =========================
    st.markdown("<br>", unsafe_allow_html=True)
   
    if st.session_state.get("generacion_error"):
        st.error(f"**Ocurrió un error durante la generación:**")
        st.error(st.session_state.generacion_error)
        st.info("💡 **Sugerencia:** Verifique los datos ingresados e intente nuevamente.")
       
    elif st.session_state.get("generacion_finalizada"):        
        st.warning("⚠️ **Recordatorio importante:** Revise y valide todo el contenido generado antes de su uso.")        
   

    # =========================
    # BOTÓN VOLVER
    # =========================
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔙 Volver al Formulario", type="primary", use_container_width=True):
            # Limpiar SOLO estados de generación, mantener los datos del formulario
            for k in ["generacion_en_ejecucion", "generacion_finalizada", "generacion_error",
                      "mostrar_pantalla_generacion", "input_data_generacion", "estado_generacion", "nombre_propuesta"]:
                st.session_state.pop(k, None)
            st.rerun()


def formulario_user_oficial(input_form_user):
    if ambiente == 'GCAAS':
        logo_base64 = get_base64_image("app/archivos/Logo PwC.png")
       
    if ambiente == 'LOCAL':
        logo_base64 = get_base64_image("./archivos/Logo PwC.png")
   
    st.set_page_config(
        page_title="Propuestas Automatizadas con GenAI | PwC Chile",
        page_icon=f"data:image/png;base64,{logo_base64}",
        layout="wide"  # Necesario para el sidebar
    )  

    # =========================
    # VERIFICAR SI DEBE MOSTRAR PANTALLA DE GENERACIÓN
    # =========================
    if st.session_state.get("mostrar_pantalla_generacion", False):
        pantalla_generacion_propuesta(st.session_state.get("input_data_generacion", {}), logo_base64)
        st.stop()  # DETENER COMPLETAMENTE LA EJECUCIÓN DEL RESTO DEL CÓDIGO

    # =========================
    # POPUP: VALIDACIÓN
    # =========================
    @st.dialog("⚠️ Campos Incompletos", width="small")
    def popup_error_validacion(campos_vacios):
        st.error("**Complete los siguientes campos obligatorios:**")
        for campo in campos_vacios:
            st.markdown(f"- **{campo}**")
        if st.button("Entendido", use_container_width=True, type="primary"):
            st.session_state.mostrar_popup_validacion = False
            st.rerun()    

    # Mostrar popup de validación si está activo
    if st.session_state.get("mostrar_popup_validacion", False):
        popup_error_validacion(st.session_state.get("campos_vacios", []))
   
    # =========================
    # ESTILOS CSS
    # =========================
    st.markdown(
        f"""
        <style>
        .logo-container {{
            position: absolute;
            top: 30px;
            right: 0px;
            z-index: 100;
        }}
        header, [data-testid="stHeader"] {{
            display: none !important;
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            height: 100%;
            width:100%;
            background-color: #DFE3E6 !important;
        }}
        .block-container {{
            max-width: 95%;
            padding: 2% 0.5%;
            background-color: transparent;
        }}
        .stColumns {{
            display: flex;
            flex-wrap: wrap;
            padding: 0rem 0.5rem;

        }}
        .stColumn {{
            flex: 1 1 300px;
            min-width: 300px;
            margin-bottom: 1rem;
            padding: 2% 2% 0;
        }}
        /* Estilos para el sidebar */
        [data-testid="stSidebar"] {{
            background-color: #f8f9fa;
            border-right: 2px solid #FE7C39;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem;
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )
 
    # Banner superior
    st.markdown(
        """
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;          
            background-color: #FE7C39;
            color: white;
            padding: 0rem 2rem;
            z-index: 1000;

            display: flex;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h1 style="margin: 0; font-size: 1.5rem;">Propuestas Automatizadas con GenAI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # INICIALIZACIÓN DE ESTADOS
    # =========================
    if "generando_propuesta" not in st.session_state:
        st.session_state.generando_propuesta = False

    # Inicializar campos SOLO si no existen
    campos_default = {
        "sub_los": "",
        "titulo_propuesta": "",
        "tipo_servicio": "",
        "palabras_claves": "",
        "socio": "",
        "cliente": "",
        "industria": "",
        "fecha_presentacion": date.today(),

        "solicitado_por": "",
        "problema": "",
        "objetivo_general": "",
        "objetivo_secundario": "",
        "alcance_funcional": "",
        "alcance_tecnico": "",
        "alcance_geo_org": "",
        "alcance_limitaciones": "",
        "equipo_responsable": [],
        "equipo_cvs": [],
        "diferenciadores": []
    }

    for campo, valor_default in campos_default.items():
        if campo not in st.session_state:
            st.session_state[campo] = valor_default

    # =========================
    # SIDEBAR - CARGA DE DOCUMENTOS
    # =========================
    with st.sidebar:
        extracted_content = render_document_sidebar(logo_base64)
    
    # Aplicar contenido extraído si se solicitó
    if st.session_state.get("apply_extracted_content", False):
        content = st.session_state.doc_consolidated_content
        apply_extracted_content_to_form(content)
        st.session_state.apply_extracted_content = False
        
        # Contar campos completados
        campos_completados = []
        if content and content.titulo_propuesta:
            campos_completados.append("Título")
        if content and content.cliente:
            campos_completados.append("Cliente")
        if content and content.palabras_claves:
            campos_completados.append("Palabras clave")
        if content and content.industria:
            campos_completados.append("Industria")
        if content and content.problema:
            campos_completados.append("Problema")
        if content and content.objetivo_general:
            campos_completados.append("Objetivo general")
        if content and content.objetivos_secundarios:
            campos_completados.append("Objetivos secundarios")
        if content and content.alcance_funcional:
            campos_completados.append("Alcance funcional")
        if content and content.alcance_tecnico:
            campos_completados.append("Alcance técnico")
        if content and content.alcance_geografico:
            campos_completados.append("Alcance geográfico")
        if content and content.limitaciones:
            campos_completados.append("Limitaciones")
        
        # Mostrar mensaje con campos completados
        if campos_completados:
            st.toast(f"✅ Formulario autocompletado: {len(campos_completados)} campos", icon="✨")
        st.rerun()

    # =========================
    # FORMULARIO PRINCIPAL
    # =========================
    st.title("Información de la Propuesta")
    st.warning("⚠️ NO UTILICE EL CONTENIDO GENERADO SIN SUPERVISIÓN Y VALIDACIÓN HUMANA")
   
    # ===== SECCIÓN 1: Información General =====
    st.subheader("📋 Paso 1: Información General")
    col1, col2 = st.columns(2)

    with col1:
        # Selección de SubLoS
        sublos_sorted = sorted(SUBLOS)
        sublos_index = sublos_sorted.index(st.session_state.sub_los) + 1 if st.session_state.sub_los in sublos_sorted else 0
        selected_sublos = st.selectbox(
            "Seleccionar Sub-LoS *",
            [""] + sublos_sorted,
            index=sublos_index,
            help="📌 Seleccione la línea de servicio específica (Sub-LoS) a la que pertenece esta propuesta."
        )
        if selected_sublos != st.session_state.sub_los:
            st.session_state.sub_los = selected_sublos
            st.session_state.tipo_servicio = ""  # Reset tipo servicio cuando cambia sublos

        titulo = st.text_input(
            "Ingresar Título de la propuesta *",
            value=st.session_state.titulo_propuesta,
            help="✏️ Ingrese un título descriptivo y profesional que resuma claramente la propuesta."
        )
        if titulo != st.session_state.titulo_propuesta:
            st.session_state.titulo_propuesta = titulo

        servicios_disponibles = sorted(SERVICIOS_POR_SUBLOS.get(st.session_state.sub_los, []))

        servicio_index = servicios_disponibles.index(st.session_state.tipo_servicio) + 1 if st.session_state.tipo_servicio in servicios_disponibles else 0
        selected_servicio = st.selectbox(
            "Seleccionar Tipo de Servicio *",
            [""] + servicios_disponibles,
            index=servicio_index,
            help="🔧 Especifique el tipo de servicio que se ofrecerá al cliente."
        )
        if selected_servicio != st.session_state.tipo_servicio:
            st.session_state.tipo_servicio = selected_servicio

        palabras = st.text_area(
            "Palabras Claves",
            value=st.session_state.palabras_claves,
            height=80,
            help="🏷️ Ingrese términos clave separados por comas."
        )
        if palabras != st.session_state.palabras_claves:
            st.session_state.palabras_claves = palabras
       
        diferenciadores_sorted = sorted(DIFERENCIADORES)
        selected_dif = st.multiselect(                
            "Diferenciadores",
            diferenciadores_sorted,
            default=st.session_state.diferenciadores,
            placeholder='Selecciona una o más opciones...',
            help="⭐ Seleccione los elementos diferenciadores de PwC."

        )
        if selected_dif != st.session_state.diferenciadores:
            st.session_state.diferenciadores = selected_dif

    with col2:
        # Filtrar socios
        socios = sorted([p["nombre"] for sublos, personas in PERSONAS_POR_SUBLOS.items()
                        for p in personas if p["rango"].lower() == "socio"])
        socio_index = socios.index(st.session_state.socio) + 1 if st.session_state.socio in socios else 0
        selected_socio = st.selectbox(
            "Seleccionar Socio *",
            [""] + socios,
            index=socio_index,
            help="👤 Seleccione el socio responsable."
        )
        if selected_socio != st.session_state.socio:
            st.session_state.socio = selected_socio
       
        cliente = st.text_input(
            "Ingresar Cliente *",
            value=st.session_state.cliente,
            help="🏢 Ingrese el nombre completo de la organización cliente."
        )
        if cliente != st.session_state.cliente:
            st.session_state.cliente = cliente

        industrias_sorted = sorted(INDUSTRIAS)
        industria_index = industrias_sorted.index(st.session_state.industria) + 1 if st.session_state.industria in industrias_sorted else 0
        selected_industria = st.selectbox(
            "Industria *",
            [""] + industrias_sorted,
            index=industria_index,
            help="🏭 Seleccione el sector industrial."
        )
        if selected_industria != st.session_state.industria:
            st.session_state.industria = selected_industria
       
        fecha = st.date_input(
            "Seleccionar Fecha de presentación",
            value=st.session_state.fecha_presentacion,
            help="📅 Indique la fecha estimada de presentación."
        )
        if fecha != st.session_state.fecha_presentacion:
            st.session_state.fecha_presentacion = fecha

        solicitado = st.text_input(
            "Solicitado por (Ingresa tu nombre y apellido) *",
            value=st.session_state.solicitado_por,
            help="👨‍💼 Ingrese el nombre de quien solicita (OBLIGATORIO)."
        )
        if solicitado != st.session_state.solicitado_por:
            st.session_state.solicitado_por = solicitado

    st.divider()
   
    # ===== SECCIÓN 2: Contexto y Alcance =====
    st.subheader("📝 Paso 2: Contexto y Alcance")
    st.info("💡 **Sugerencia:** Sea específico y detallado en cada campo.")
   
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### 🎯 Contexto del Proyecto")
       
        problema = st.text_area(
            "Problema *",
            height=100,
            value=st.session_state.problema,
            help="🔍 Describa claramente el problema que enfrenta el cliente."
        )
        if problema != st.session_state.problema:
            st.session_state.problema = problema
       
        obj_general = st.text_area(
            "Objetivo General *",
            height=100,
            value=st.session_state.objetivo_general,
            help="🎯 Defina el objetivo principal a alcanzar."
        )

        if obj_general != st.session_state.objetivo_general:
            st.session_state.objetivo_general = obj_general
       
        obj_secundario = st.text_area(
            "Objetivo Secundario *",
            height=100,
            value=st.session_state.objetivo_secundario,
            help="📋 Especifique objetivos complementarios."
        )
        if obj_secundario != st.session_state.objetivo_secundario:
            st.session_state.objetivo_secundario = obj_secundario

    with col4:
        st.markdown("##### 📐 Alcance del Proyecto")
       
        alc_func = st.text_area(
            "Funcional",
            height=100,
            value=st.session_state.alcance_funcional,
            help="⚙️ Detalle las funcionalidades específicas."
        )
        if alc_func != st.session_state.alcance_funcional:
            st.session_state.alcance_funcional = alc_func
       
        alc_tec = st.text_area(
            "Técnico",
            height=100,

            value=st.session_state.alcance_tecnico,
            help="💻 Especifique los aspectos técnicos."
        )
        if alc_tec != st.session_state.alcance_tecnico:
            st.session_state.alcance_tecnico = alc_tec
       
        alc_geo = st.text_area(
            "Geográfico y Organizacional",
            height=100,
            value=st.session_state.alcance_geo_org,
            help="🌍 Indique el alcance geográfico y organizacional."
        )
        if alc_geo != st.session_state.alcance_geo_org:
            st.session_state.alcance_geo_org = alc_geo
       
        alc_lim = st.text_area(
            "Limitaciones",
            height=100,
            value=st.session_state.alcance_limitaciones,
            help="⚠️ Describa lo que NO está incluido en el alcance."
        )
        if alc_lim != st.session_state.alcance_limitaciones:
            st.session_state.alcance_limitaciones = alc_lim

    st.divider()
   
    # =========================

    # BOTÓN GENERAR PROPUESTA
    # =========================
    st.subheader("🚀 Generar Propuesta")
    st.info("📋 **Antes de generar:** Revise que todos los campos obligatorios (*) estén completos.")

    def sincronizar_y_armar_input():
        # Crear diccionario con datos directamente del session_state
        input_data = {
            'sublos': st.session_state.sub_los,
            'socio': st.session_state.socio,
            'titulo_propuesta': st.session_state.titulo_propuesta,
            'cliente': st.session_state.cliente,
            'tipo_servicio': st.session_state.tipo_servicio,
            'industria': st.session_state.industria,
            'solicitado_por': st.session_state.solicitado_por,
            'fecha_presentacion': st.session_state.fecha_presentacion,
            'diferenciadores': ', '.join(st.session_state.diferenciadores),
            'palabras_claves': st.session_state.palabras_claves,
            'problema': st.session_state.problema,
            'objetivo_general': st.session_state.objetivo_general,
            'objetivo_secundario': st.session_state.objetivo_secundario,
            'alcance_funcional': st.session_state.alcance_funcional,
            'alcance_tecnico': st.session_state.alcance_tecnico,
            'alcance_geografico': st.session_state.alcance_geo_org,
            'limitaciones': st.session_state.alcance_limitaciones,
            'equipo': "",

            'cv': "",
        }
        return input_data

    def validar_obligatorios(input_data):
        campos_vacios = []
        if not input_data['sublos']:
            campos_vacios.append("Sub-LoS")
        if not input_data['titulo_propuesta']:
            campos_vacios.append("Título de propuesta")
        if not input_data['cliente']:
            campos_vacios.append("Cliente")
        if not input_data['problema']:
            campos_vacios.append("Problema")
        if not input_data['objetivo_general']:
            campos_vacios.append("Objetivo general")
        if not input_data['solicitado_por']:
            campos_vacios.append("Solicitado por")
        return campos_vacios

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Generar Propuesta", use_container_width=True, type="primary", key="btn_generar_propuesta",
                    help="Haga clic para iniciar la generación (2-5 minutos)."):
            data = sincronizar_y_armar_input()
            faltantes = validar_obligatorios(data)

            if faltantes:
                st.session_state.campos_vacios = faltantes
                st.session_state.mostrar_popup_validacion = True
                st.rerun()
            else:
                # Guardar datos y mostrar pantalla de generación
                st.session_state.input_data_generacion = data
                st.session_state.mostrar_pantalla_generacion = True
                st.rerun()