import lancedb
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory


@dataclass
class SearchConfig:
    """Configuración para búsqueda vectorial combinada"""
    db_uri: str
    cv_table_name: str
    credentials_table_name: str
    openai_api_key: str
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4"
    temperature: float = 0.3
    top_k_cvs: int = 5
    top_k_credentials: int = 8


@dataclass
class SearchContext:
    """Contexto de búsqueda para mantener información entre consultas"""
    last_cv_results: List[Dict] = field(default_factory=list)
    last_credential_results: List[Dict] = field(default_factory=list)
    selected_cvs: List[str] = field(default_factory=list)  # emails
    selected_credentials: List[int] = field(default_factory=list)  # IDs
    search_history: List[str] = field(default_factory=list)
    current_industry: Optional[str] = None
    current_skills: List[str] = field(default_factory=list)
    conversation_memory: Optional[ConversationBufferMemory] = None


class CombinedSearchAgent:
    """
    Agente especializado en búsqueda combinada de CVs y Credenciales
    con análisis integrado y mantenimiento de contexto
    """
    
    def __init__(self, config: SearchConfig):
        """
        Inicializa el agente de búsqueda combinada
        
        Args:
            config: Configuración de búsqueda
        """
        self.config = config
        self.context = SearchContext()
        
        # Inicializar modelo de embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.openai_api_key,
            model=config.embedding_model
        )
        
        # Inicializar LLM para el agente
        self.llm = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model=config.llm_model,
            temperature=config.temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Conectar a la base de datos
        self.db = lancedb.connect(config.db_uri)
        self.cv_table = self.db.open_table(config.cv_table_name)
        self.credentials_table = self.db.open_table(config.credentials_table_name)
        
        # Inicializar memoria de conversación
        self.context.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Configurar prompts del agente
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Configura el prompt para análisis combinado de CVs y Credenciales"""
        
        combined_system_template = """Eres un especialista experto en análisis integral de talento y credenciales empresariales.
Tu rol es proporcionar una visión completa y estratégica combinando perfiles profesionales con casos de éxito relevantes.

Contexto de búsqueda actual:
- Búsqueda: {query}
- Candidatos encontrados: {cv_count}
- Credenciales encontradas: {cred_count}
- Industria objetivo: {current_industry}
- Habilidades clave: {current_skills}
- Historial de búsquedas: {search_history}

INSTRUCCIONES PARA EL ANÁLISIS:

1. **SECCIÓN DE PERFILES/CVs**:
   Para cada candidato relevante, proporciona:
   - Nombre completo y cargo actual
   - Email de contacto
   - Por qué es relevante para esta búsqueda
   - Principales fortalezas y experiencia
   - Nivel de coincidencia (Alto/Medio/Bajo)

2. **SECCIÓN DE CREDENCIALES** - Presenta en formato de TARJETAS:
   ╔══════════════════════════════════════╗
   ║ 🏆 CREDENCIAL #X                      ║
   ╠══════════════════════════════════════╣
   ║ 📋 Proyecto: [Nombre]                 ║
   ║ 🏢 Cliente: [Cliente/Confidencial]    ║
   ║ 🏭 Industria: [Industria]             ║
   ║ 📅 Período: [Año inicio - fin]        ║
   ║ ⭐ Relevancia: [Alta/Media/Baja]      ║
   ╠══════════════════════════════════════╣
   ║ 💡 Problema/Contexto:                 ║
   ║ [Resumen del problema]                ║
   ║                                       ║
   ║ 🔧 Solución Implementada:             ║
   ║ [Resumen de la solución]              ║
   ║                                       ║
   ║ 📊 Entregables Clave:                 ║
   ║ • [Punto 1]                           ║
   ║ • [Punto 2]                           ║
   ║                                       ║
   ║ 👥 Equipo Involucrado:                ║
   ║ [Resumen del equipo]                  ║
   ╚══════════════════════════════════════╝

3. **RECOMENDACIONES ESTRATÉGICAS**:
   - Equipo core recomendado (3-5 personas clave)
   - Credenciales más relevantes para mostrar al cliente      

Sé específico, usa los datos disponibles y proporciona insights accionables."""

        # Configurar prompt combinado
        self.combined_system_prompt = SystemMessagePromptTemplate.from_template(combined_system_template)
        
        combined_human_template = """Consulta del usuario: {query}

================================================================================
PERFILES/CVs ENCONTRADOS (Top {cv_count}):
================================================================================
{candidates}

================================================================================
CREDENCIALES ENCONTRADAS (Top {cred_count}):
================================================================================
{credentials}

Por favor proporciona un análisis completo e integrado siguiendo las instrucciones del sistema."""
        
        self.combined_human_prompt = HumanMessagePromptTemplate.from_template(combined_human_template)
        self.combined_chat_prompt = ChatPromptTemplate.from_messages([
            self.combined_system_prompt,
            self.combined_human_prompt
        ])
        
        # Crear chain para análisis combinado
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.combined_chat_prompt,
            verbose=False
        )
    
    def _format_candidate(self, record: pd.Series, score: float) -> str:
        """Formatea la información de un candidato para el análisis"""
        metadata = {}
        if 'metadata' in record and record['metadata']:
            try:
                metadata = json.loads(record['metadata'])
            except:
                pass
        
        candidate_info = []
        candidate_info.append(f"**Candidato:** {record.get('nombres', 'N/A')} {record.get('apellido_paterno', '')} {record.get('apellido_materno', '')}")
        candidate_info.append(f"**Email:** {metadata.get('email', record.get('email', 'N/A'))}")
        candidate_info.append(f"**Cargo Actual:** {record.get('cargo', 'N/A')}")
        candidate_info.append(f"**Línea de Servicio:** {record.get('los', 'N/A')} - {record.get('sublos', 'N/A')}")
        candidate_info.append(f"**Similitud:** {score:.2%}")
        
        if record.get('antecedente'):
            candidate_info.append(f"\n**Resumen Profesional:**\n{record['antecedente'][:500]}...")
        
        if record.get('habilidades'):
            candidate_info.append(f"\n**Habilidades:**\n{record['habilidades'][:300]}...")
        
        if record.get('areas_experiencia'):
            candidate_info.append(f"\n**Áreas de Experiencia:**\n{record['areas_experiencia'][:200]}...")
        
        if record.get('industrias'):
            candidate_info.append(f"\n**Industrias:**\n{record['industrias']}")
        
        if record.get('educacion'):
            candidate_info.append(f"\n**Educación:**\n{record['educacion'][:300]}...")
        
        return "\n".join(candidate_info)
    
    def _format_credential(self, record: pd.Series, score: float) -> str:
        """Formatea la información de una credencial para el análisis"""
        credential_info = []
        
        credential_info.append(f"**ID Credencial:** {record.get('id', 'N/A')}")
        credential_info.append(f"**Servicio:** {record.get('nombre_del_servicio', 'N/A')}")
        credential_info.append(f"**Cliente:** {record.get('cliente', record.get('cliente_anon', 'Confidencial'))}")
        credential_info.append(f"**Industria:** {record.get('industria', 'N/A')} - {record.get('industria_l2', '')}")
        credential_info.append(f"**País:** {record.get('pais', 'N/A')}")
        credential_info.append(f"**Período:** {record.get('ano_de_inicio', 'N/A')} - {record.get('ano_de_cierre', 'N/A')}")
        credential_info.append(f"**Área:** {record.get('area', 'N/A')}")
        credential_info.append(f"**Similitud:** {score:.2%}")
        
        if record.get('problema_contexto'):
            credential_info.append(f"\n**Problema/Contexto:**\n{record['problema_contexto'][:400]}...")
        
        if record.get('solucion'):
            credential_info.append(f"\n**Solución:**\n{record['solucion'][:400]}...")
        
        if record.get('entregables'):
            credential_info.append(f"\n**Entregables:**\n{record['entregables'][:300]}...")
        
        if record.get('socio'):
            credential_info.append(f"\n**Socio:** {record['socio']}")
        
        if record.get('gerente_servicio'):
            credential_info.append(f"**Gerente de Servicio:** {record['gerente_servicio']}")
        
        if record.get('equipo_pwc'):
            credential_info.append(f"\n**Equipo PwC:**\n{record['equipo_pwc'][:200]}...")
        
        return "\n".join(credential_info)
    
    def _vector_search(self, query: str, table, top_k: int) -> List[Dict[str, Any]]:
        """Realiza búsqueda vectorial genérica en una tabla"""
        # Generar embedding de la consulta
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array(query_embedding)
        
        # Obtener registros
        records_df = table.to_pandas()
        
        # Calcular similitudes
        similarities = []
        for idx in records_df.index:
            record_vector = np.array(records_df.loc[idx, 'vector'])
            cosine_sim = np.dot(query_vector, record_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(record_vector)
            )
            similarities.append({
                'index': idx,
                'score': float(cosine_sim),
                'record': records_df.loc[idx]
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def search_and_analyze(self, query: str, top_k_cvs: Optional[int] = None, top_k_creds: Optional[int] = None) -> Dict[str, Any]:
        """
        Realiza búsqueda combinada de CVs y Credenciales con análisis integrado
        
        Args:
            query: Consulta del usuario
            top_k_cvs: Número de CVs a retornar (default: config.top_k_cvs)
            top_k_creds: Número de credenciales a retornar (default: config.top_k_credentials)
            
        Returns:
            Diccionario con resultados combinados y análisis integrado
        """
        if top_k_cvs is None:
            top_k_cvs = self.config.top_k_cvs
        if top_k_creds is None:
            top_k_creds = self.config.top_k_credentials
        
        print("="*80)
        print("🔄 BÚSQUEDA COMBINADA: CVs + CREDENCIALES")
        print("="*80)
        print(f"📝 Consulta: {query}")
        print("-"*80)
        
        # Actualizar contexto
        self.context.search_history.append(query)
        self._extract_context_from_query(query)
        
        # Buscar CVs
        print(f"\n🔍 Buscando CVs (Top {top_k_cvs})...")
        cv_results = self._vector_search(query, self.cv_table, top_k_cvs)
        self.context.last_cv_results = cv_results
        print(f"✅ Encontrados {len(cv_results)} candidatos")
        
        # Buscar Credenciales
        print(f"\n🔍 Buscando Credenciales (Top {top_k_creds})...")
        cred_results = self._vector_search(query, self.credentials_table, top_k_creds)
        self.context.last_credential_results = cred_results
        print(f"✅ Encontradas {len(cred_results)} credenciales")
        
        # Formatear resultados para el análisis
        formatted_candidates = []
        for i, result in enumerate(cv_results, 1):
            print(f"\n📋 Procesando Candidato {i}/{len(cv_results)}")
            formatted = self._format_candidate(result['record'], result['score'])
            formatted_candidates.append(f"\n{'='*40}\nCANDIDATO {i}\n{'='*40}\n{formatted}")
        
        formatted_credentials = []
        for i, result in enumerate(cred_results, 1):
            print(f"\n🏆 Procesando Credencial {i}/{len(cred_results)}")
            formatted = self._format_credential(result['record'], result['score'])
            formatted_credentials.append(f"\n{'='*40}\nCREDENCIAL {i}\n{'='*40}\n{formatted}")
        
        candidates_text = "\n".join(formatted_candidates) if formatted_candidates else "No se encontraron candidatos"
        credentials_text = "\n".join(formatted_credentials) if formatted_credentials else "No se encontraron credenciales"
        
        # Ejecutar análisis integrado
        print("\n" + "="*80)
        print("🤖 GENERANDO ANÁLISIS INTEGRADO")
        print("="*80)
        print("\n")
        
        analysis = self.analysis_chain.run(
            query=query,
            candidates=candidates_text,
            credentials=credentials_text,
            cv_count=len(cv_results),
            cred_count=len(cred_results),
            current_industry=self.context.current_industry or "No especificada",
            current_skills=", ".join(self.context.current_skills) if self.context.current_skills else "No especificadas",
            search_history="; ".join(self.context.search_history[-5:]) if self.context.search_history else "Primera búsqueda"
        )
        
        # Preparar respuesta estructurada
        response = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'cvs': {
                'total': len(cv_results),
                'results': [
                    {
                        'name': f"{r['record'].get('nombres', '')} {r['record'].get('apellido_paterno', '')}".strip(),
                        'email': r['record'].get('email', ''),
                        'position': r['record'].get('cargo', ''),
                        'skills': r['record'].get('habilidades', ''),
                        'industries': r['record'].get('industrias', ''),
                        'score': r['score']
                    }
                    for r in cv_results
                ]
            },
            'credentials': {
                'total': len(cred_results),
                'results': [
                    {
                        'id': int(r['record'].get('id', 0)),
                        'service': r['record'].get('nombre_del_servicio', ''),
                        'client': r['record'].get('cliente', r['record'].get('cliente_anon', 'Confidencial')),
                        'industry': r['record'].get('industria', ''),
                        'industry_l2': r['record'].get('industria_l2', ''),
                        'country': r['record'].get('pais', ''),
                        'year_start': r['record'].get('ano_de_inicio', ''),
                        'year_end': r['record'].get('ano_de_cierre', ''),
                        'problem': r['record'].get('problema_contexto', ''),
                        'solution': r['record'].get('solucion', ''),
                        'deliverables': r['record'].get('entregables', ''),
                        'team': r['record'].get('equipo_pwc', ''),
                        'score': r['score']
                    }
                    for r in cred_results
                ]
            },
            'analysis': analysis,
            'context': {
                'industry': self.context.current_industry,
                'skills': self.context.current_skills,
                'search_count': len(self.context.search_history),
                'recent_searches': self.context.search_history[-5:]
            }
        }
        
        print("\n" + "="*80)
        print("✅ ANÁLISIS COMBINADO COMPLETADO")
        print("="*80)
        
        return response
    
    def select_cv(self, email: str) -> Dict[str, Any]:
        """Selecciona un CV por email para incluir en exportación"""
        if email not in self.context.selected_cvs:
            self.context.selected_cvs.append(email)
            
            # Buscar el CV en los resultados
            selected_cv = None
            for result in self.context.last_cv_results:
                record = result['record']
                if record.get('email') == email:
                    selected_cv = {
                        'name': f"{record.get('nombres', '')} {record.get('apellido_paterno', '')}".strip(),
                        'email': email,
                        'position': record.get('cargo', ''),
                        'skills': record.get('habilidades', '')
                    }
                    break
            
            return {
                'status': 'success',
                'message': f'✅ CV seleccionado: {email}',
                'selected_cv': selected_cv,
                'total_selected_cvs': len(self.context.selected_cvs)
            }
        else:
            return {
                'status': 'already_selected',
                'message': f'⚠️ El CV {email} ya estaba seleccionado',
                'total_selected_cvs': len(self.context.selected_cvs)
            }
    
    def select_credential(self, credential_id: int) -> Dict[str, Any]:
        """Selecciona una credencial por ID"""
        return self.agent.select_credential(credential_id)
    
    def get_selections(self) -> Dict[str, Any]:
        """Obtiene todas las selecciones actuales"""
        return self.agent.get_selected_items()
    
    def clear_selections(self) -> Dict[str, Any]:
        """Limpia todas las selecciones"""
        return self.agent.clear_selections()
    
    def export(self, context_proposal: str, client: str, industry: str) -> Dict[str, Any]:
        """Exporta las selecciones con el contexto proporcionado"""
        export_context = {
            'context_proposal': context_proposal,
            'client': client,
            'industry': industry
        }
        return self.agent.export_selections(export_context)
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del agente"""
        return self.agent.get_search_summary()


# ============= Ejemplo de uso =============
if __name__ == "__main__":
    # Configuración
    config = SearchConfig(
        db_uri="./data/lancedb",
        cv_table_name="personal_embeddings",
        credentials_table_name="credentials_embeddings",
        openai_api_key="sk-...",  # Tu API key
        llm_model="gpt-4",
        temperature=0.3,
        top_k_cvs=5,
        top_k_credentials=8
    )
    
    # Crear API del agente
    search_api = SearchAgentAPI(config)
    
    # Ejemplo 1: Búsqueda combinada
    print("\n" + "="*80)
    print("EJEMPLO: BÚSQUEDA COMBINADA")
    print("="*80)
    
    results = search_api.search(
        query="Necesito un equipo para proyecto de transformación digital en minería con experiencia en IA y automatización de procesos",
        top_k_cvs=5,
        top_k_credentials=8
    )
    
    print(f"\n📊 RESUMEN DE RESULTADOS:")
    print(f"   - CVs encontrados: {results['cvs']['total']}")
    print(f"   - Credenciales encontradas: {results['credentials']['total']}")
    print(f"   - Industria detectada: {results['context']['industry']}")
    print(f"   - Habilidades detectadas: {', '.join(results['context']['skills'])}")
    
    # Ejemplo 2: Seleccionar elementos
    if results['cvs']['results']:
        # Seleccionar primer CV
        first_cv = results['cvs']['results'][0]
        selection_result = search_api.select_profile(first_cv['email'])
        print(f"\n{selection_result['message']}")
    
    if results['credentials']['results']:
        # Seleccionar primera credencial
        first_cred = results['credentials']['results'][0]
        selection_result = search_api.select_credential(first_cred['id'])
        print(f"{selection_result['message']}")
    
    # Ejemplo 3: Ver selecciones
    selections = search_api.get_selections()
    print(f"\n📌 ELEMENTOS SELECCIONADOS:")
    print(f"   - CVs: {selections['total_selections']['cvs']}")
    print(f"   - Credenciales: {selections['total_selections']['credentials']}")
    
    # Ejemplo 4: Exportar
    if selections['total_selections']['cvs'] > 0 or selections['total_selections']['credentials'] > 0:
        export_result = search_api.export(
            context_proposal="Implementación de solución de IA para optimización de procesos mineros",
            client="Minera Ejemplo S.A.",
            industry="Minería"
        )
        print(f"\n{export_result['message']}")
    
    # Ejemplo 5: Ver estado general
    status = search_api.get_status()
    print(f"\n📈 ESTADO DEL AGENTE:")
    print(f"   - Búsquedas realizadas: {status['context']['search_count']}")
    print(f"   - Últimos resultados: {status['last_results']['cvs_found']} CVs, {status['last_results']['credentials_found']} Credenciales")
    print(f"   - Selecciones activas: {status['selections']['selected_cvs_count']} CVs, {status['selections']['selected_credentials_count']} Credenciales")d: int) -> Dict[str, Any]:
        """Selecciona una credencial por ID para incluir en exportación"""
        if credential_id not in self.context.selected_credentials:
            self.context.selected_credentials.append(credential_id)
            
            # Buscar la credencial en los resultados
            selected_credential = None
            for result in self.context.last_credential_results:
                record = result['record']
                if record.get('id') == credential_id:
                    selected_credential = {
                        'id': credential_id,
                        'service': record.get('nombre_del_servicio', ''),
                        'client': record.get('cliente', record.get('cliente_anon', 'Confidencial')),
                        'industry': record.get('industria', '')
                    }
                    break
            
            return {
                'status': 'success',
                'message': f'✅ Credencial seleccionada: ID {credential_id}',
                'selected_credential': selected_credential,
                'total_selected_credentials': len(self.context.selected_credentials)
            }
        else:
            return {
                'status': 'already_selected',
                'message': f'⚠️ La credencial {credential_id} ya estaba seleccionada',
                'total_selected_credentials': len(self.context.selected_credentials)
            }
    
    def get_selected_items(self) -> Dict[str, Any]:
        """Obtiene todos los items seleccionados para exportación"""
        return {
            'selected_cvs': self.context.selected_cvs,
            'selected_credentials': self.context.selected_credentials,
            'total_selections': {
                'cvs': len(self.context.selected_cvs),
                'credentials': len(self.context.selected_credentials)
            },
            'details': {
                'cv_emails': self.context.selected_cvs,
                'credential_ids': self.context.selected_credentials
            }
        }
    
    def clear_selections(self):
        """Limpia todas las selecciones"""
        self.context.selected_cvs = []
        self.context.selected_credentials = []
        return {
            'status': 'success',
            'message': '🗑️ Todas las selecciones han sido limpiadas'
        }
    
    def export_selections(self, export_context: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepara los datos seleccionados para exportación
        
        Args:
            export_context: Diccionario con context_proposal, client, industry
            
        Returns:
            Diccionario con todos los datos preparados para exportación
        """
        if not self.context.selected_cvs and not self.context.selected_credentials:
            return {
                'status': 'error',
                'message': '❌ No hay elementos seleccionados para exportar'
            }
        
        # Recopilar información completa de los CVs seleccionados
        selected_cv_details = []
        for email in self.context.selected_cvs:
            for result in self.context.last_cv_results:
                record = result['record']
                if record.get('email') == email:
                    metadata = {}
                    if 'metadata' in record and record['metadata']:
                        try:
                            metadata = json.loads(record['metadata'])
                        except:
                            pass
                    
                    selected_cv_details.append({
                        'name': f"{record.get('nombres', '')} {record.get('apellido_paterno', '')}".strip(),
                        'email': email,
                        'position': record.get('cargo', ''),
                        'los': record.get('los', ''),
                        'sublos': record.get('sublos', ''),
                        'skills': record.get('habilidades', ''),
                        'experience': record.get('antecedente', ''),
                        'industries': record.get('industrias', ''),
                        'education': record.get('educacion', ''),
                        'areas_experiencia': record.get('areas_experiencia', '')
                    })
                    break
        
        # Recopilar información completa de las credenciales seleccionadas
        selected_credential_details = []
        for cred_id in self.context.selected_credentials:
            for result in self.context.last_credential_results:
                record = result['record']
                if record.get('id') == cred_id:
                    selected_credential_details.append({
                        'id': cred_id,
                        'service': record.get('nombre_del_servicio', ''),
                        'client': record.get('cliente', record.get('cliente_anon', 'Confidencial')),
                        'industry': record.get('industria', ''),
                        'industry_l2': record.get('industria_l2', ''),
                        'country': record.get('pais', ''),
                        'area': record.get('area', ''),
                        'period': f"{record.get('ano_de_inicio', '')} - {record.get('ano_de_cierre', '')}",
                        'problem': record.get('problema_contexto', ''),
                        'solution': record.get('solucion', ''),
                        'deliverables': record.get('entregables', ''),
                        'team': record.get('equipo_pwc', ''),
                        'partner': record.get('socio', ''),
                        'service_manager': record.get('gerente_servicio', ''),
                        'contraparte': record.get('contraparte', '')
                    })
                    break
        
        export_data = {
            'export_context': export_context,
            'timestamp': datetime.now().isoformat(),
            'selected_profiles': {
                'count': len(selected_cv_details),
                'emails': self.context.selected_cvs,
                'details': selected_cv_details
            },
            'selected_credentials': {
                'count': len(selected_credential_details),
                'ids': self.context.selected_credentials,
                'details': selected_credential_details
            },
            'search_context': {
                'industry': self.context.current_industry,
                'skills': self.context.current_skills,
                'total_searches': len(self.context.search_history),
                'search_history': self.context.search_history[-10:]
            }
        }
        
        return {
            'status': 'success',
            'message': '✅ Datos preparados para exportación',
            'export_data': export_data
        }
    
    def _extract_context_from_query(self, query: str):
        """Extrae contexto (industria, habilidades) del query"""
        # Lista de industrias comunes
        industries = ['minería', 'retail', 'banca', 'tecnología', 'salud', 'educación', 
                     'manufactura', 'energía', 'telecomunicaciones', 'construcción',
                     'finanzas', 'seguros', 'logística', 'transporte', 'inmobiliaria',
                     'consumo masivo', 'farmacéutica', 'automotriz', 'agricultura']
        
        # Lista de habilidades técnicas comunes
        tech_skills = ['python', 'java', 'javascript', 'react', 'angular', 'aws', 'azure',
                      'docker', 'kubernetes', 'sql', 'mongodb', 'machine learning', 'ai',
                      'scrum', 'agile', 'devops', 'cloud', 'microservicios', 'data science',
                      'analytics', 'power bi', 'tableau', 'sap', 'oracle', 'salesforce',
                      'blockchain', 'iot', 'cybersecurity', 'rpa', 'transformación digital']
        
        query_lower = query.lower()
        
        # Buscar industrias en el query
        for industry in industries:
            if industry in query_lower:
                self.context.current_industry = industry.capitalize()
                break
        
        # Buscar habilidades en el query
        found_skills = []
        for skill in tech_skills:
            if skill in query_lower:
                found_skills.append(skill.upper() if len(skill) <= 3 else skill.capitalize())
        
        if found_skills:
            self.context.current_skills = found_skills
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado actual de búsqueda y selecciones"""
        return {
            'context': {
                'current_industry': self.context.current_industry,
                'current_skills': self.context.current_skills,
                'search_count': len(self.context.search_history),
                'recent_searches': self.context.search_history[-5:] if self.context.search_history else []
            },
            'last_results': {
                'cvs_found': len(self.context.last_cv_results),
                'credentials_found': len(self.context.last_credential_results)
            },
            'selections': {
                'selected_cvs_count': len(self.context.selected_cvs),
                'selected_credentials_count': len(self.context.selected_credentials),
                'cv_emails': self.context.selected_cvs,
                'credential_ids': self.context.selected_credentials
            }
        }


# ============= Clase de integración para uso externo =============
class SearchAgentAPI:
    """
    API simplificada para usar el agente de búsqueda combinada
    """
    
    def __init__(self, config: SearchConfig):
        self.agent = CombinedSearchAgent(config)
    
    def search(self, query: str, top_k_cvs: int = 5, top_k_credentials: int = 8) -> Dict[str, Any]:
        """Realiza una búsqueda combinada"""
        return self.agent.search_and_analyze(query, top_k_cvs, top_k_credentials)
    
    def select_profile(self, email: str) -> Dict[str, Any]:
        """Selecciona un perfil/CV por email"""
        return self.agent.select_cv(email)
    
    def select_credential(self, credential_i