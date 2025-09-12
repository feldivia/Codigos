import lancedb
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import LanceDB
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler


# ====== NUEVA ADICIÓN: Enum para tipos de búsqueda ======
class SearchType(Enum):
    CV = "cv"
    CREDENTIALS = "credentials"
    UNKNOWN = "unknown"


@dataclass
class SearchConfig:
    """Configuración para búsqueda vectorial"""
    db_uri: str
    table_name: str
    openai_api_key: str
    # ====== NUEVA ADICIÓN: Tabla para credenciales ======
    credentials_table_name: str = "credentials_embeddings"
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4"
    temperature: float = 0.3
    top_k: int = 5


# ====== NUEVA CLASE: Router para clasificar consultas ======
class QueryRouter:
    """Router que clasifica las consultas según su intención"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_router_prompt()
    
    def _setup_router_prompt(self):
        """Configura el prompt para clasificación de consultas"""
        router_template = """Eres un clasificador de consultas. Tu trabajo es determinar si una consulta busca:
1. PERFILES/CVs: Búsquedas sobre personas, candidatos, desarrolladores, profesionales, experiencia laboral, habilidades técnicas, roles, cargos, etc.
2. CREDENCIALES: Búsquedas sobre certificaciones, títulos académicos, diplomas, acreditaciones, licencias profesionales, cursos completados, etc.

Analiza la siguiente consulta y responde ÚNICAMENTE con una de estas palabras:
- CV (si busca perfiles o información de curriculums)
- CREDENTIALS (si busca credenciales o certificaciones)
- UNKNOWN (si no está claro)

Consulta: {query}

Clasificación:"""
        
        self.router_prompt = ChatPromptTemplate.from_template(router_template)
        self.router_chain = LLMChain(
            llm=self.llm,
            prompt=self.router_prompt,
            verbose=False
        )
    
    def classify(self, query: str) -> SearchType:
        """Clasifica una consulta"""
        result = self.router_chain.run(query=query).strip().upper()
        
        if "CV" in result:
            return SearchType.CV
        elif "CREDENTIALS" in result:
            return SearchType.CREDENTIALS
        else:
            return SearchType.UNKNOWN


# ====== CLASE MODIFICADA: Ahora maneja múltiples tipos de búsqueda ======
class UniversalSearchAgent:
    """Agente universal que puede buscar tanto CVs como credenciales"""
    
    def __init__(self, config: SearchConfig):
        """Inicializa el agente de búsqueda universal"""
        self.config = config
        
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
        
        # ====== NUEVA ADICIÓN: Inicializar router ======
        self.router = QueryRouter(self.llm)
        
        # Conectar a la base de datos
        self.db = lancedb.connect(config.db_uri)
        
        # ====== MODIFICACIÓN: Abrir ambas tablas ======
        self.cv_table = self.db.open_table(config.table_name)
        self.credentials_table = self.db.open_table(config.credentials_table_name)
        
        # Configurar prompts para cada tipo de búsqueda
        self._setup_cv_prompts()
        self._setup_credentials_prompts()
    
    def _setup_cv_prompts(self):
        """Configura los prompts del agente especialista en CVs"""
        
        system_template = """Eres un especialista experto en análisis de currículums y reclutamiento de talento tecnológico.
Tu rol es analizar perfiles profesionales y proporcionar recomendaciones precisas basadas en los requisitos del usuario.

Cuando analices los candidatos, considera:
1. **Experiencia Técnica**: Habilidades, tecnologías y herramientas dominadas
2. **Experiencia Profesional**: Roles, responsabilidades y logros
3. **Educación y Certificaciones**: Formación académica y certificaciones relevantes
4. **Industrias**: Sectores donde ha trabajado
5. **Fit Cultural**: Basado en el tipo de empresas y proyectos previos

Para cada candidato recomendado, proporciona:
- Nombre completo y cargo actual
- Resumen de por qué es relevante para la búsqueda
- Principales fortalezas que lo hacen destacar
- Experiencia específica relacionada con los requisitos
- Nivel de coincidencia (Alto/Medio/Bajo) con explicación

Sé específico y utiliza la información disponible de los CVs para justificar tus recomendaciones.
Si no hay candidatos que cumplan bien con los criterios, sé honesto al respecto y sugiere alternativas."""

        human_template = """Consulta del usuario: {query}

Candidatos encontrados:
{candidates}

Por favor, analiza estos perfiles y proporciona recomendaciones detalladas basadas en la consulta."""

        self.cv_system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        self.cv_human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        self.cv_chat_prompt = ChatPromptTemplate.from_messages([
            self.cv_system_prompt,
            self.cv_human_prompt
        ])
        
        self.cv_chain = LLMChain(
            llm=self.llm,
            prompt=self.cv_chat_prompt,
            verbose=False
        )
    
    # ====== NUEVA ADICIÓN: Prompts para credenciales ======
    def _setup_credentials_prompts(self):
        """Configura los prompts para análisis de credenciales"""
        
        system_template = """Eres un especialista en análisis de credenciales y certificaciones profesionales.
Tu rol es evaluar y recomendar profesionales basándote en sus credenciales académicas y certificaciones.

Cuando analices las credenciales, considera:
1. **Certificaciones Profesionales**: Certificados técnicos, licencias, acreditaciones
2. **Formación Académica**: Títulos universitarios, postgrados, doctorados
3. **Vigencia**: Estado actual de las certificaciones (vigentes/expiradas)
4. **Relevancia**: Qué tan relacionadas están con los requisitos
5. **Nivel**: Nivel de especialización (básico, intermedio, avanzado, experto)

Para cada profesional con credenciales relevantes, proporciona:
- Nombre completo
- Lista de credenciales relevantes con fechas
- Nivel de especialización
- Relevancia para la búsqueda
- Estado de vigencia de certificaciones"""

        human_template = """Consulta del usuario: {query}

Credenciales encontradas:
{credentials}

Por favor, analiza estas credenciales y proporciona recomendaciones detalladas."""

        self.credentials_system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        self.credentials_human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        self.credentials_chat_prompt = ChatPromptTemplate.from_messages([
            self.credentials_system_prompt,
            self.credentials_human_prompt
        ])
        
        self.credentials_chain = LLMChain(
            llm=self.llm,
            prompt=self.credentials_chat_prompt,
            verbose=False
        )
    
    def _format_candidate(self, record: pd.Series, score: float) -> str:
        """Formatea la información de un candidato para el análisis"""
        # Extraer metadata
        metadata = {}
        if 'metadata' in record and record['metadata']:
            try:
                metadata = json.loads(record['metadata'])
            except:
                pass
        
        # Construir información del candidato
        candidate_info = []
        candidate_info.append(f"**Candidato:** {record.get('nombres', 'N/A')} {record.get('apellido_paterno', '')} {record.get('apellido_materno', '')}")
        candidate_info.append(f"**Email:** {metadata.get('email', record.get('email', 'N/A'))}")
        candidate_info.append(f"**Cargo Actual:** {record.get('cargo', 'N/A')}")
        candidate_info.append(f"**Línea de Servicio:** {record.get('los', 'N/A')} - {record.get('sublos', 'N/A')}")
        candidate_info.append(f"**Similitud:** {score:.2%}")
        
        # Añadir antecedentes si existen
        if record.get('antecedente'):
            candidate_info.append(f"\n**Resumen Profesional:**\n{record['antecedente'][:500]}...")
        
        # Añadir habilidades
        if record.get('habilidades'):
            candidate_info.append(f"\n**Habilidades:**\n{record['habilidades'][:300]}...")
        
        # Añadir áreas de experiencia
        if record.get('areas_experiencia'):
            candidate_info.append(f"\n**Áreas de Experiencia:**\n{record['areas_experiencia'][:200]}...")
        
        # Añadir industrias
        if record.get('industrias'):
            candidate_info.append(f"\n**Industrias:**\n{record['industrias']}")
        
        # Añadir educación
        if record.get('educacion'):
            candidate_info.append(f"\n**Educación:**\n{record['educacion'][:300]}...")
        
        return "\n".join(candidate_info)
    
    # ====== NUEVO MÉTODO: Formatear credenciales ======
    def _format_credential(self, record: pd.Series, score: float) -> str:
        """Formatea la información de una credencial"""
        credential_info = []
        credential_info.append(f"**Titular:** {record.get('nombre_completo', 'N/A')}")
        credential_info.append(f"**Credencial:** {record.get('credencial', 'N/A')}")
        credential_info.append(f"**Institución:** {record.get('institucion', 'N/A')}")
        credential_info.append(f"**Fecha:** {record.get('fecha', 'N/A')}")
        credential_info.append(f"**Estado:** {record.get('estado', 'Vigente')}")
        credential_info.append(f"**Similitud:** {score:.2%}")
        
        if record.get('descripcion'):
            credential_info.append(f"\n**Descripción:**\n{record['descripcion'][:300]}...")
        
        return "\n".join(credential_info)
    
    # ====== MÉTODO REFACTORIZADO: Búsqueda vectorial genérica ======
    def _vector_search(self, query: str, table: Any, top_k: int) -> List[Dict[str, Any]]:
        """Realiza búsqueda vectorial en una tabla específica"""
        
        # Generar embedding de la consulta
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array(query_embedding)
        
        # Obtener todos los registros
        records_df = table.to_pandas()
        
        # Calcular similitudes
        similarities = []
        
        for idx in records_df.index:
            # Obtener vector del registro
            record_vector = np.array(records_df.loc[idx, 'vector'])
            
            # Calcular similitud coseno
            cosine_sim = np.dot(query_vector, record_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(record_vector)
            )
            
            similarities.append({
                'index': idx,
                'score': float(cosine_sim),
                'record': records_df.loc[idx]
            })
        
        # Ordenar por similitud y tomar top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    # ====== MÉTODO PRINCIPAL MODIFICADO: Ahora decide qué búsqueda hacer ======
    def search_and_analyze(self, query: str, top_k: Optional[int] = None, force_type: Optional[SearchType] = None) -> Dict[str, Any]:
        """
        Realiza búsqueda inteligente decidiendo el tipo según el contexto
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a analizar
            force_type: Forzar un tipo específico de búsqueda (opcional)
            
        Returns:
            Diccionario con resultados y análisis del agente
        """
        if top_k is None:
            top_k = self.config.top_k
        
        print("="*80)
        print("🤖 AGENTE DE BÚSQUEDA INTELIGENTE")
        print("="*80)
        
        # ====== NUEVA LÓGICA: Clasificar la consulta ======
        if force_type:
            search_type = force_type
            print(f"🎯 Tipo de búsqueda forzado: {search_type.value}")
        else:
            print("🔍 Analizando contexto de la consulta...")
            search_type = self.router.classify(query)
            print(f"📊 Tipo de búsqueda detectado: {search_type.value}")
        
        # ====== NUEVA LÓGICA: Ejecutar búsqueda según el tipo ======
        if search_type == SearchType.CV:
            return self._search_cvs(query, top_k)
        elif search_type == SearchType.CREDENTIALS:
            return self._search_credentials(query, top_k)
        else:
            # Si no está claro, buscar en ambos
            print("⚠️ Tipo no claro, buscando en ambas fuentes...")
            cv_results = self._search_cvs(query, top_k//2)
            cred_results = self._search_credentials(query, top_k//2)
            return self._merge_results(cv_results, cred_results)
    
    # ====== MÉTODO REFACTORIZADO: Búsqueda de CVs ======
    def _search_cvs(self, query: str, top_k: int) -> Dict[str, Any]:
        """Búsqueda específica en CVs"""
        print("\n📋 Buscando en base de datos de CVs...")
        print("-" * 60)
        
        # Realizar búsqueda vectorial
        search_results = self._vector_search(query, self.cv_table, top_k)
        
        if not search_results:
            return {
                'query': query,
                'search_type': 'cv',
                'candidates': [],
                'analysis': "No se encontraron candidatos que coincidan con la búsqueda."
            }
        
        print(f"✅ Encontrados {len(search_results)} candidatos")
        
        # Formatear candidatos
        formatted_candidates = []
        for i, result in enumerate(search_results, 1):
            print(f"\n📋 Candidato {i}:")
            print("-" * 40)
            formatted = self._format_candidate(result['record'], result['score'])
            print(formatted[:500] + "...\n")
            formatted_candidates.append(f"\n--- CANDIDATO {i} ---\n{formatted}")
        
        candidates_text = "\n".join(formatted_candidates)
        
        # Análisis con el agente
        print("="*80)
        print("🔍 ANÁLISIS DEL ESPECIALISTA EN CVs")
        print("="*80)
        print("\n")
        
        analysis = self.cv_chain.run(
            query=query,
            candidates=candidates_text
        )
        
        print("\n" + "="*80)
        print("✅ Análisis de CVs completado")
        print("="*80)
        
        return {
            'query': query,
            'search_type': 'cv',
            'total_candidates': len(search_results),
            'candidates': [
                {
                    'name': f"{r['record'].get('nombres', '')} {r['record'].get('apellido_paterno', '')}".strip(),
                    'email': r['record'].get('email', ''),
                    'position': r['record'].get('cargo', ''),
                    'score': r['score'],
                    'summary': r['record'].get('antecedente', '')[:200] if r['record'].get('antecedente') else ''
                }
                for r in search_results
            ],
            'analysis': analysis
        }
    
    # ====== NUEVO MÉTODO: Búsqueda de credenciales ======
    def _search_credentials(self, query: str, top_k: int) -> Dict[str, Any]:
        """Búsqueda específica en credenciales"""
        print("\n🎓 Buscando en base de datos de credenciales...")
        print("-" * 60)
        
        # Realizar búsqueda vectorial
        search_results = self._vector_search(query, self.credentials_table, top_k)
        
        if not search_results:
            return {
                'query': query,
                'search_type': 'credentials',
                'credentials': [],
                'analysis': "No se encontraron credenciales que coincidan con la búsqueda."
            }
        
        print(f"✅ Encontradas {len(search_results)} credenciales")
        
        # Formatear credenciales
        formatted_credentials = []
        for i, result in enumerate(search_results, 1):
            print(f"\n🎓 Credencial {i}:")
            print("-" * 40)
            formatted = self._format_credential(result['record'], result['score'])
            print(formatted[:500] + "...\n")
            formatted_credentials.append(f"\n--- CREDENCIAL {i} ---\n{formatted}")
        
        credentials_text = "\n".join(formatted_credentials)
        
        # Análisis con el agente
        print("="*80)
        print("🔍 ANÁLISIS DEL ESPECIALISTA EN CREDENCIALES")
        print("="*80)
        print("\n")
        
        analysis = self.credentials_chain.run(
            query=query,
            credentials=credentials_text
        )
        
        print("\n" + "="*80)
        print("✅ Análisis de credenciales completado")
        print("="*80)
        
        return {
            'query': query,
            'search_type': 'credentials',
            'total_credentials': len(search_results),
            'credentials': [
                {
                    'holder': result['record'].get('nombre_completo', ''),
                    'credential': result['record'].get('credencial', ''),
                    'institution': result['record'].get('institucion', ''),
                    'date': result['record'].get('fecha', ''),
                    'score': result['score']
                }
                for result in search_results
            ],
            'analysis': analysis
        }
    
    # ====== NUEVO MÉTODO: Combinar resultados de ambas búsquedas ======
    def _merge_results(self, cv_results: Dict, cred_results: Dict) -> Dict[str, Any]:
        """Combina resultados de CVs y credenciales"""
        return {
            'query': cv_results['query'],
            'search_type': 'mixed',
            'cv_results': cv_results,
            'credential_results': cred_results,
            'analysis': f"**Resultados de Perfiles:**\n{cv_results['analysis']}\n\n**Resultados de Credenciales:**\n{cred_results['analysis']}"
        }
    
    def search_by_skills(self, skills: List[str], top_k: Optional[int] = None) -> Dict[str, Any]:
        """Busca candidatos por habilidades específicas"""
        query = f"Busco profesionales con experiencia en: {', '.join(skills)}"
        return self.search_and_analyze(query, top_k)
    
    def search_by_role(self, role: str, requirements: str = "", top_k: Optional[int] = None) -> Dict[str, Any]:
        """Busca candidatos para un rol específico"""
        query = f"Necesito candidatos para el rol de {role}."
        if requirements:
            query += f" Requisitos importantes: {requirements}"
        return self.search_and_analyze(query, top_k)
    
    def search_by_experience(self, industry: str, years: int, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Busca candidatos por experiencia en industria"""
        query = f"Busco profesionales con al menos {years} años de experiencia en {industry}"
        return self.search_and_analyze(query, top_k)


def quick_search(
    query: str,
    db_uri: str,
    table_name: str,
    credentials_table_name: str,
    openai_api_key: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Función rápida para realizar búsquedas sin instanciar la clase
    
    Args:
        query: Consulta del usuario
        db_uri: URI de la base de datos
        table_name: Nombre de la tabla de CVs
        credentials_table_name: Nombre de la tabla de credenciales
        openai_api_key: API Key de OpenAI
        top_k: Número de resultados
        
    Returns:
        Diccionario con resultados y análisis
    """
    config = SearchConfig(
        db_uri=db_uri,
        table_name=table_name,
        credentials_table_name=credentials_table_name,
        openai_api_key=openai_api_key,
        top_k=top_k
    )
    
    agent = UniversalSearchAgent(config)
    return agent.search_and_analyze(query, top_k)
