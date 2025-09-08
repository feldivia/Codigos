"""
Sistema de Agentes con LangGraph para gestión de CVs y Credenciales
Requiere: pip install langgraph langchain-openai langchain-core pydantic typing-extensions httpx
"""

from typing import TypedDict, Annotated, List, Dict, Optional, Literal
from typing_extensions import TypedDict
from enum import Enum
import httpx
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ============= Configuración =============
class Config:
    """Configuración central del sistema"""
    OPENAI_API_KEY = "your-api-key"
    CVS_API_URL = "https://api.example.com/cvs"
    CREDENTIALS_API_URL = "https://api.example.com/credentials"
    EXPORT_API_URL = "https://api.example.com/export"
    MODEL = "gpt-4-turbo-preview"


# ============= Modelos de Datos =============
class IntentType(str, Enum):
    """Tipos de intención del usuario"""
    SEARCH_CVS = "search_cvs"
    SEARCH_CREDENTIALS = "search_credentials"
    EXPORT = "export"
    SELECT_PROFILE = "select_profile"
    SELECT_CREDENTIAL = "select_credential"
    GENERAL = "general"
    UNCLEAR = "unclear"


class ExportData(BaseModel):
    """Datos requeridos para exportación"""
    context_proposal: str = Field(description="Contexto de la propuesta")
    client: str = Field(description="Nombre del cliente")
    industry: str = Field(description="Industria del cliente")


class ProfileInfo(BaseModel):
    """Información de un perfil/CV"""
    id: str
    name: str
    email: str
    skills: List[str]
    experience: str
    selected: bool = False


class CredentialInfo(BaseModel):
    """Información de una credencial"""
    id: str
    type: str
    description: str
    validity: str
    selected: bool = False


# ============= Estado del Agente =============
class AgentState(TypedDict):
    """Estado completo del agente que persiste durante la conversación"""
    messages: List[BaseMessage]
    current_intent: Optional[IntentType]
    last_search_results: Dict
    
    # Datos acumulados
    selected_emails: List[str]
    selected_credential_ids: List[str]
    selected_profiles: List[ProfileInfo]
    selected_credentials: List[CredentialInfo]
    
    # Datos de exportación
    export_data: Optional[Dict]
    export_ready: bool
    
    # Control de flujo
    next_action: Optional[str]
    error: Optional[str]
    
    # Contexto de búsqueda
    search_query: Optional[str]
    awaiting_export_info: bool


# ============= APIs Externas =============
class ExternalAPIClient:
    """Cliente para llamar a las APIs externas"""
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
    
    async def search_cvs(self, query: str) -> List[ProfileInfo]:
        """Busca en la API de CVs"""
        try:
            response = self.client.post(
                Config.CVS_API_URL,
                json={"query": query, "limit": 10}
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                ProfileInfo(
                    id=item["id"],
                    name=item["name"],
                    email=item["email"],
                    skills=item.get("skills", []),
                    experience=item.get("experience", "")
                )
                for item in data.get("results", [])
            ]
        except Exception as e:
            print(f"Error en API de CVs: {e}")
            return []
    
    async def search_credentials(self, query: str) -> List[CredentialInfo]:
        """Busca en la API de credenciales"""
        try:
            response = self.client.post(
                Config.CREDENTIALS_API_URL,
                json={"query": query, "limit": 10}
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                CredentialInfo(
                    id=item["id"],
                    type=item["type"],
                    description=item["description"],
                    validity=item.get("validity", "")
                )
                for item in data.get("results", [])
            ]
        except Exception as e:
            print(f"Error en API de Credenciales: {e}")
            return []
    
    async def export_data(self, export_payload: Dict) -> Dict:
        """Exporta los datos seleccionados"""
        try:
            response = self.client.post(
                Config.EXPORT_API_URL,
                json=export_payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error en API de Exportación: {e}")
            return {"error": str(e)}


# ============= Nodos del Grafo =============
class GraphNodes:
    """Contiene todos los nodos del grafo de LangGraph"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL,
            temperature=0,
            api_key=Config.OPENAI_API_KEY
        )
        self.api_client = ExternalAPIClient()
    
    def classify_intent(self, state: AgentState) -> AgentState:
        """Clasifica la intención del usuario basándose en su mensaje"""
        
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Si estamos esperando información de exportación
        if state.get("awaiting_export_info", False):
            state["current_intent"] = IntentType.EXPORT
            return state
        
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Clasifica la intención del usuario en una de estas categorías:
            - search_cvs: Si busca perfiles, CVs, candidatos, personas
            - search_credentials: Si busca credenciales, certificaciones, acreditaciones
            - select_profile: Si quiere seleccionar un perfil específico (menciona nombre o número)
            - select_credential: Si quiere seleccionar una credencial específica
            - export: Si quiere exportar, generar reporte, enviar datos
            - general: Conversación general o preguntas
            
            Responde SOLO con la categoría, sin explicaciones."""),
            ("human", "{message}")
        ])
        
        chain = intent_prompt | self.llm
        response = chain.invoke({"message": last_message})
        
        intent_map = {
            "search_cvs": IntentType.SEARCH_CVS,
            "search_credentials": IntentType.SEARCH_CREDENTIALS,
            "select_profile": IntentType.SELECT_PROFILE,
            "select_credential": IntentType.SELECT_CREDENTIAL,
            "export": IntentType.EXPORT,
            "general": IntentType.GENERAL
        }
        
        intent_str = response.content.strip().lower()
        state["current_intent"] = intent_map.get(intent_str, IntentType.UNCLEAR)
        
        return state
    
    async def search_cvs_node(self, state: AgentState) -> AgentState:
        """Busca en la API de CVs"""
        last_message = state["messages"][-1].content
        
        # Extraer query de búsqueda
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extrae los términos de búsqueda relevantes del mensaje del usuario para buscar CVs/perfiles."),
            ("human", "{message}")
        ])
        
        chain = query_prompt | self.llm
        query_response = chain.invoke({"message": last_message})
        search_query = query_response.content
        
        # Llamar a la API
        profiles = await self.api_client.search_cvs(search_query)
        
        # Guardar resultados
        state["last_search_results"] = {
            "type": "cvs",
            "results": [p.dict() for p in profiles],
            "timestamp": datetime.now().isoformat()
        }
        
        # Construir mensaje de respuesta
        if profiles:
            response_text = f"Encontré {len(profiles)} perfiles:\n\n"
            for i, profile in enumerate(profiles, 1):
                response_text += f"{i}. **{profile.name}**\n"
                response_text += f"   - Email: {profile.email}\n"
                response_text += f"   - Skills: {', '.join(profile.skills[:5])}\n"
                response_text += f"   - Experiencia: {profile.experience[:100]}...\n\n"
            response_text += "\nPuedes seleccionar perfiles escribiendo su número o nombre."
        else:
            response_text = "No encontré perfiles que coincidan con tu búsqueda."
        
        state["messages"].append(AIMessage(content=response_text))
        return state
    
    async def search_credentials_node(self, state: AgentState) -> AgentState:
        """Busca en la API de credenciales"""
        last_message = state["messages"][-1].content
        
        # Extraer query de búsqueda
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extrae los términos de búsqueda relevantes del mensaje para buscar credenciales/certificaciones."),
            ("human", "{message}")
        ])
        
        chain = query_prompt | self.llm
        query_response = chain.invoke({"message": last_message})
        search_query = query_response.content
        
        # Llamar a la API
        credentials = await self.api_client.search_credentials(search_query)
        
        # Guardar resultados
        state["last_search_results"] = {
            "type": "credentials",
            "results": [c.dict() for c in credentials],
            "timestamp": datetime.now().isoformat()
        }
        
        # Construir mensaje de respuesta
        if credentials:
            response_text = f"Encontré {len(credentials)} credenciales:\n\n"
            for i, cred in enumerate(credentials, 1):
                response_text += f"{i}. **{cred.type}**\n"
                response_text += f"   - ID: {cred.id}\n"
                response_text += f"   - Descripción: {cred.description}\n"
                response_text += f"   - Validez: {cred.validity}\n\n"
            response_text += "\nPuedes seleccionar credenciales escribiendo su número o tipo."
        else:
            response_text = "No encontré credenciales que coincidan con tu búsqueda."
        
        state["messages"].append(AIMessage(content=response_text))
        return state
    
    def select_profile_node(self, state: AgentState) -> AgentState:
        """Maneja la selección de perfiles"""
        last_message = state["messages"][-1].content
        
        if not state.get("last_search_results") or state["last_search_results"]["type"] != "cvs":
            state["messages"].append(
                AIMessage(content="Primero necesitas buscar perfiles antes de seleccionarlos.")
            )
            return state
        
        # Extraer número o nombre del perfil seleccionado
        selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extrae el número o nombre del perfil que el usuario quiere seleccionar.
            Si es un número, responde solo con el número.
            Si es un nombre, responde con el nombre exacto."""),
            ("human", "{message}")
        ])
        
        chain = selection_prompt | self.llm
        selection = chain.invoke({"message": last_message}).content.strip()
        
        profiles = state["last_search_results"]["results"]
        selected_profile = None
        
        # Buscar por número o nombre
        try:
            index = int(selection) - 1
            if 0 <= index < len(profiles):
                selected_profile = profiles[index]
        except ValueError:
            # Buscar por nombre
            for profile in profiles:
                if selection.lower() in profile["name"].lower():
                    selected_profile = profile
                    break
        
        if selected_profile:
            # Agregar a seleccionados
            if selected_profile["email"] not in state["selected_emails"]:
                state["selected_emails"].append(selected_profile["email"])
                state["selected_profiles"].append(ProfileInfo(**selected_profile))
                
                response = f"✅ Perfil seleccionado: {selected_profile['name']} ({selected_profile['email']})\n"
                response += f"Total de perfiles seleccionados: {len(state['selected_emails'])}"
            else:
                response = f"Este perfil ya está seleccionado."
        else:
            response = "No pude identificar el perfil. Intenta con el número o nombre exacto."
        
        state["messages"].append(AIMessage(content=response))
        return state
    
    def select_credential_node(self, state: AgentState) -> AgentState:
        """Maneja la selección de credenciales"""
        last_message = state["messages"][-1].content
        
        if not state.get("last_search_results") or state["last_search_results"]["type"] != "credentials":
            state["messages"].append(
                AIMessage(content="Primero necesitas buscar credenciales antes de seleccionarlas.")
            )
            return state
        
        # Similar lógica para seleccionar credenciales
        selection_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extrae el número o tipo de credencial que el usuario quiere seleccionar."),
            ("human", "{message}")
        ])
        
        chain = selection_prompt | self.llm
        selection = chain.invoke({"message": last_message}).content.strip()
        
        credentials = state["last_search_results"]["results"]
        selected_cred = None
        
        try:
            index = int(selection) - 1
            if 0 <= index < len(credentials):
                selected_cred = credentials[index]
        except ValueError:
            for cred in credentials:
                if selection.lower() in cred["type"].lower():
                    selected_cred = cred
                    break
        
        if selected_cred:
            if selected_cred["id"] not in state["selected_credential_ids"]:
                state["selected_credential_ids"].append(selected_cred["id"])
                state["selected_credentials"].append(CredentialInfo(**selected_cred))
                
                response = f"✅ Credencial seleccionada: {selected_cred['type']} (ID: {selected_cred['id']})\n"
                response += f"Total de credenciales seleccionadas: {len(state['selected_credential_ids'])}"
            else:
                response = f"Esta credencial ya está seleccionada."
        else:
            response = "No pude identificar la credencial. Intenta con el número o tipo exacto."
        
        state["messages"].append(AIMessage(content=response))
        return state
    
    def prepare_export_node(self, state: AgentState) -> AgentState:
        """Prepara la exportación solicitando datos adicionales"""
        
        # Verificar si ya tenemos los datos de exportación
        if state.get("export_data") and all(
            state["export_data"].get(k) for k in ["context_proposal", "client", "industry"]
        ):
            state["export_ready"] = True
            return state
        
        # Extraer información del último mensaje
        last_message = state["messages"][-1].content
        
        if state.get("awaiting_export_info", False):
            # Intentar extraer los datos del mensaje
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extrae la siguiente información del mensaje del usuario:
                - context_proposal: Contexto o descripción de la propuesta
                - client: Nombre del cliente
                - industry: Industria o sector
                
                Responde en formato JSON con estas tres claves. Si falta información, usa null."""),
                ("human", "{message}")
            ])
            
            chain = extraction_prompt | self.llm
            response = chain.invoke({"message": last_message})
            
            try:
                export_info = json.loads(response.content)
                
                # Verificar qué información falta
                missing = []
                if not export_info.get("context_proposal"):
                    missing.append("contexto de la propuesta")
                if not export_info.get("client"):
                    missing.append("nombre del cliente")
                if not export_info.get("industry"):
                    missing.append("industria")
                
                if missing:
                    response_text = f"Necesito los siguientes datos para exportar:\n"
                    response_text += "\n".join(f"- {item}" for item in missing)
                    response_text += "\n\nPor favor proporciónalos."
                    state["awaiting_export_info"] = True
                else:
                    state["export_data"] = export_info
                    state["export_ready"] = True
                    state["awaiting_export_info"] = False
                    response_text = "✅ Datos de exportación completos. Procesando..."
                
                state["messages"].append(AIMessage(content=response_text))
                
            except json.JSONDecodeError:
                state["messages"].append(
                    AIMessage(content="Por favor proporciona el contexto de propuesta, cliente e industria.")
                )
                state["awaiting_export_info"] = True
        else:
            # Primera vez que solicita exportar
            response_text = "Para exportar los datos seleccionados, necesito la siguiente información:\n"
            response_text += "1. **Contexto de la propuesta**: ¿Para qué es esta propuesta?\n"
            response_text += "2. **Cliente**: ¿Cuál es el nombre del cliente?\n"
            response_text += "3. **Industria**: ¿En qué industria o sector opera?\n\n"
            response_text += f"Actualmente tienes seleccionados:\n"
            response_text += f"- {len(state['selected_emails'])} perfiles\n"
            response_text += f"- {len(state['selected_credential_ids'])} credenciales\n"
            
            state["messages"].append(AIMessage(content=response_text))
            state["awaiting_export_info"] = True
        
        return state
    
    async def execute_export_node(self, state: AgentState) -> AgentState:
        """Ejecuta la exportación llamando a la API externa"""
        
        if not state.get("export_ready"):
            return state
        
        # Preparar payload para la API
        export_payload = {
            "export_data": state["export_data"],
            "selected_profiles": [
                {
                    "email": p.email,
                    "name": p.name,
                    "skills": p.skills
                }
                for p in state["selected_profiles"]
            ],
            "selected_credentials": [
                {
                    "id": c.id,
                    "type": c.type,
                    "description": c.description
                }
                for c in state["selected_credentials"]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Llamar a la API de exportación
        result = await self.api_client.export_data(export_payload)
        
        if "error" not in result:
            response_text = "✅ **Exportación exitosa!**\n\n"
            response_text += f"Se exportaron:\n"
            response_text += f"- {len(state['selected_emails'])} perfiles\n"
            response_text += f"- {len(state['selected_credential_ids'])} credenciales\n\n"
            response_text += f"**Detalles de la exportación:**\n"
            response_text += f"- Cliente: {state['export_data']['client']}\n"
            response_text += f"- Industria: {state['export_data']['industry']}\n"
            response_text += f"- Contexto: {state['export_data']['context_proposal']}\n"
            
            if result.get("export_id"):
                response_text += f"\n**ID de exportación:** {result['export_id']}"
            
            # Limpiar estado de exportación
            state["export_data"] = None
            state["export_ready"] = False
            state["awaiting_export_info"] = False
        else:
            response_text = f"❌ Error en la exportación: {result['error']}"
        
        state["messages"].append(AIMessage(content=response_text))
        return state
    
    def general_response_node(self, state: AgentState) -> AgentState:
        """Maneja respuestas generales y proporciona ayuda"""
        
        # Generar una respuesta contextual
        context = f"""
        Eres un asistente para gestión de CVs y credenciales.
        
        Estado actual:
        - Perfiles seleccionados: {len(state['selected_emails'])}
        - Credenciales seleccionadas: {len(state['selected_credential_ids'])}
        
        Puedes ayudar con:
        1. Buscar perfiles/CVs
        2. Buscar credenciales
        3. Seleccionar elementos de las búsquedas
        4. Exportar los datos seleccionados
        """
        
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", context),
            ("human", "{message}")
        ])
        
        chain = general_prompt | self.llm
        response = chain.invoke({
            "message": state["messages"][-1].content if state["messages"] else "Hola"
        })
        
        state["messages"].append(AIMessage(content=response.content))
        return state


# ============= Construcción del Grafo =============
def build_agent_graph():
    """Construye el grafo de LangGraph con todos los nodos y edges"""
    
    # Inicializar el grafo con el estado
    workflow = StateGraph(AgentState)
    
    # Crear instancia de los nodos
    nodes = GraphNodes()
    
    # Agregar nodos al grafo
    workflow.add_node("classify_intent", nodes.classify_intent)
    workflow.add_node("search_cvs", nodes.search_cvs_node)
    workflow.add_node("search_credentials", nodes.search_credentials_node)
    workflow.add_node("select_profile", nodes.select_profile_node)
    workflow.add_node("select_credential", nodes.select_credential_node)
    workflow.add_node("prepare_export", nodes.prepare_export_node)
    workflow.add_node("execute_export", nodes.execute_export_node)
    workflow.add_node("general_response", nodes.general_response_node)
    
    # Definir el punto de entrada
    workflow.set_entry_point("classify_intent")
    
    # Función para determinar el siguiente nodo basado en la intención
    def route_by_intent(state: AgentState) -> str:
        """Router que decide el siguiente nodo basado en la intención"""
        
        intent = state.get("current_intent")
        
        # Si estamos listos para exportar
        if state.get("export_ready") and intent == IntentType.EXPORT:
            return "execute_export"
        
        # Routing basado en intención
        intent_routes = {
            IntentType.SEARCH_CVS: "search_cvs",
            IntentType.SEARCH_CREDENTIALS: "search_credentials",
            IntentType.SELECT_PROFILE: "select_profile",
            IntentType.SELECT_CREDENTIAL: "select_credential",
            IntentType.EXPORT: "prepare_export",
            IntentType.GENERAL: "general_response",
            IntentType.UNCLEAR: "general_response",
        }
        
        return intent_routes.get(intent, "general_response")
    
    # Agregar edges condicionales
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "search_cvs": "search_cvs",
            "search_credentials": "search_credentials",
            "select_profile": "select_profile",
            "select_credential": "select_credential",
            "prepare_export": "prepare_export",
            "execute_export": "execute_export",
            "general_response": "general_response",
        }
    )
    
    # Todos los nodos terminan el flujo (vuelven a esperar input del usuario)
    workflow.add_edge("search_cvs", END)
    workflow.add_edge("search_credentials", END)
    workflow.add_edge("select_profile", END)
    workflow.add_edge("select_credential", END)
    workflow.add_edge("prepare_export", END)
    workflow.add_edge("execute_export", END)
    workflow.add_edge("general_response", END)
    
    # Compilar el grafo con checkpointer para mantener estado
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph


# ============= Clase Principal del Agente =============
class LangGraphAgent:
    """Agente principal que maneja la conversación"""
    
    def __init__(self):
        self.graph = build_agent_graph()
        self.thread_id = "default"  # Puedes hacer esto dinámico por sesión
    
    def get_initial_state(self) -> AgentState:
        """Retorna el estado inicial del agente"""
        return {
            "messages": [],
            "current_intent": None,
            "last_search_results": {},
            "selected_emails": [],
            "selected_credential_ids": [],
            "selected_profiles": [],
            "selected_credentials": [],
            "export_data": None,
            "export_ready": False,
            "next_action": None,
            "error": None,
            "search_query": None,
            "awaiting_export_info": False
        }
    
    async def process_message(self, user_message: str, thread_id: str = None) -> str:
        """
        Procesa un mensaje del usuario y retorna la respuesta del agente
        
        Args:
            user_message: Mensaje del usuario
            thread_id: ID único de la conversación (para mantener estado)
        
        Returns:
            Respuesta del agente
        """
        if thread_id:
            self.thread_id = thread_id
        
        # Preparar el estado con el nuevo mensaje
        input_state = {
            "messages": [HumanMessage(content=user_message)]
        }
        
        # Ejecutar el grafo
        config = {"configurable": {"thread_id": self.thread_id}}
        result = await self.graph.ainvoke(input_state, config)
        
        # Extraer la última respuesta del agente
        if result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    return msg.content
        
        return "No pude procesar tu mensaje. Por favor intenta de nuevo."
    
    def get_current_state(self, thread_id: str = None) -> Dict:
        """
        Obtiene el estado actual de la conversación
        
        Returns:
            Estado actual con información sobre selecciones
        """
        if thread_id:
            self.thread_id = thread_id
            
        config = {"configurable": {"thread_id": self.thread_id}}
        state = self.graph.get_state(config)
        
        if state and state.values:
            return {
                "selected_emails": state.values.get("selected_emails", []),
                "selected_credentials": state.values.get("selected_credential_ids", []),
                "total_profiles": len(state.values.get("selected_profiles", [])),
                "total_credentials": len(state.values.get("selected_credentials", [])),
                "awaiting_export_info": state.values.get("awaiting_export_info", False)
            }
        return {
            "selected_emails": [],
            "selected_credentials": [],
            "total_profiles": 0,
            "total_credentials": 0,
            "awaiting_export_info": False
        }
    
    def clear_selections(self, thread_id: str = None):
        """Limpia todas las selecciones manteniendo el historial de conversación"""
        if thread_id:
            self.thread_id = thread_id
            
        config = {"configurable": {"thread_id": self.thread_id}}
        current_state = self.graph.get_state(config)
        
        if current_state and current_state.values:
            # Actualizar solo las selecciones
            updated_state = current_state.values.copy()
            updated_state["selected_emails"] = []
            updated_state["selected_credential_ids"] = []
            updated_state["selected_profiles"] = []
            updated_state["selected_credentials"] = []
            
            # Actualizar el estado
            self.graph.update_state(config, updated_state)


# ============= Integración con Streamlit =============
class StreamlitIntegration:
    """
    Clase para integrar el agente con Streamlit
    Uso en tu app de Streamlit:
    
    import streamlit as st
    from langgraph_agent import StreamlitIntegration
    
    # Inicializar el agente
    if 'agent_integration' not in st.session_state:
        st.session_state.agent_integration = StreamlitIntegration()
    
    # Procesar mensaje
    user_input = st.text_input("Tu mensaje:")
    if user_input:
        response = await st.session_state.agent_integration.handle_message(
            user_input, 
            st.session_state.get('session_id', 'default')
        )
        st.write(response)
    
    # Mostrar estado
    state_info = st.session_state.agent_integration.get_state_summary()
    st.sidebar.write(f"Perfiles seleccionados: {state_info['total_profiles']}")
    st.sidebar.write(f"Credenciales seleccionadas: {state_info['total_credentials']}")
    """
    
    def __init__(self):
        self.agent = LangGraphAgent()
    
    async def handle_message(self, message: str, session_id: str = "default") -> str:
        """Maneja un mensaje del usuario y retorna la respuesta"""
        return await self.agent.process_message(message, session_id)
    
    def get_state_summary(self, session_id: str = "default") -> Dict:
        """Obtiene un resumen del estado actual"""
        return self.agent.get_current_state(session_id)
    
    def clear_all_selections(self, session_id: str = "default"):
        """Limpia todas las selecciones"""
        self.agent.clear_selections(session_id)
    
    def format_response_markdown(self, response: str) -> str:
        """Formatea la respuesta para mejor visualización en Streamlit"""
        # Agregar formato especial para elementos seleccionados
        response = response.replace("✅", "✅ **")
        response = response.replace("❌", "❌ **")
        return response


# ============= Ejemplo de Uso con Streamlit =============
"""
# archivo: app_streamlit.py

import streamlit as st
import asyncio
from langgraph_agent import StreamlitIntegration

# Configuración de página
st.set_page_config(
    page_title="Gestor de CVs y Credenciales",
    page_icon="📋",
    layout="wide"
)

# Inicializar sesión
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().timestamp()}"

if 'agent' not in st.session_state:
    st.session_state.agent = StreamlitIntegration()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Título principal
st.title("🤖 Asistente de CVs y Credenciales")

# Sidebar con información del estado
with st.sidebar:
    st.header("📊 Estado Actual")
    
    # Obtener estado
    state = st.session_state.agent.get_state_summary(st.session_state.session_id)
    
    # Mostrar métricas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Perfiles", state['total_profiles'])
    with col2:
        st.metric("Credenciales", state['total_credentials'])
    
    # Lista de seleccionados
    if state['selected_emails']:
        st.subheader("📧 Emails Seleccionados")
        for email in state['selected_emails']:
            st.text(f"• {email}")
    
    if state['selected_credentials']:
        st.subheader("🔐 IDs de Credenciales")
        for cred_id in state['selected_credentials']:
            st.text(f"• {cred_id}")
    
    # Botón para limpiar selecciones
    if st.button("🗑️ Limpiar Selecciones"):
        st.session_state.agent.clear_all_selections(st.session_state.session_id)
        st.rerun()
    
    # Indicador de estado de exportación
    if state['awaiting_export_info']:
        st.warning("⏳ Esperando información para exportar...")

# Área principal de chat
st.header("💬 Conversación")

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu mensaje aquí..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Obtener respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            # Ejecutar de forma asíncrona
            response = asyncio.run(
                st.session_state.agent.handle_message(
                    prompt, 
                    st.session_state.session_id
                )
            )
            
            # Formatear y mostrar respuesta
            formatted_response = st.session_state.agent.format_response_markdown(response)
            st.markdown(formatted_response)
    
    # Agregar respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": response})

# Información de ayuda
with st.expander("ℹ️ Cómo usar este asistente"):
    st.markdown('''
    ### Comandos disponibles:
    
    **Buscar perfiles/CVs:**
    - "Busca perfiles de desarrolladores Python"
    - "Necesito CVs de project managers"
    
    **Buscar credenciales:**
    - "Busca credenciales de AWS"
    - "Muéstrame certificaciones de Scrum"
    
    **Seleccionar elementos:**
    - "Selecciona el perfil 1"
    - "Quiero la credencial de Azure"
    
    **Exportar datos:**
    - "Exporta los datos seleccionados"
    - "Genera un reporte con las selecciones"
    
    El sistema te pedirá información adicional cuando sea necesario.
    ''')

# Footer
st.markdown("---")
st.caption("Sistema de gestión inteligente con LangGraph")
"""


# ============= Utilidades Adicionales =============
class AgentUtils:
    """Utilidades para el manejo del agente"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Valida formato de email"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        return re.match(pattern, email) is not None
    
    @staticmethod
    def format_profile_summary(profiles: List[ProfileInfo]) -> str:
        """Genera un resumen formateado de perfiles"""
        if not profiles:
            return "No hay perfiles seleccionados"
        
        summary = f"### 📋 Resumen de {len(profiles)} Perfiles Seleccionados\n\n"
        for i, profile in enumerate(profiles, 1):
            summary += f"**{i}. {profile.name}**\n"
            summary += f"   - Email: {profile.email}\n"
            summary += f"   - Skills principales: {', '.join(profile.skills[:3])}\n"
            summary += f"   - Experiencia: {profile.experience[:50]}...\n\n"
        
        return summary
    
    @staticmethod
    def format_credential_summary(credentials: List[CredentialInfo]) -> str:
        """Genera un resumen formateado de credenciales"""
        if not credentials:
            return "No hay credenciales seleccionadas"
        
        summary = f"### 🔐 Resumen de {len(credentials)} Credenciales Seleccionadas\n\n"
        for i, cred in enumerate(credentials, 1):
            summary += f"**{i}. {cred.type}**\n"
            summary += f"   - ID: {cred.id}\n"
            summary += f"   - Descripción: {cred.description}\n"
            summary += f"   - Validez: {cred.validity}\n\n"
        
        return summary
    
    @staticmethod
    def export_to_json(state: AgentState) -> str:
        """Exporta el estado actual a formato JSON"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "profiles": [
                {
                    "name": p.name,
                    "email": p.email,
                    "skills": p.skills,
                    "experience": p.experience
                }
                for p in state.get("selected_profiles", [])
            ],
            "credentials": [
                {
                    "id": c.id,
                    "type": c.type,
                    "description": c.description,
                    "validity": c.validity
                }
                for c in state.get("selected_credentials", [])
            ],
            "summary": {
                "total_profiles": len(state.get("selected_profiles", [])),
                "total_credentials": len(state.get("selected_credentials", [])),
                "emails": state.get("selected_emails", []),
                "credential_ids": state.get("selected_credential_ids", [])
            }
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)


# ============= Configuración de Logging =============
import logging

def setup_logging():
    """Configura el sistema de logging para debugging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('langgraph_agent.log'),
            logging.StreamHandler()
        ]
    )
    
    # Logger específico para el agente
    logger = logging.getLogger('LangGraphAgent')
    logger.setLevel(logging.DEBUG)
    
    return logger


# ============= Testing y Debugging =============
class AgentTester:
    """Clase para testing del agente"""
    
    @staticmethod
    async def test_flow():
        """Test básico del flujo completo"""
        agent = LangGraphAgent()
        test_session = "test_session_001"
        
        # Test de búsqueda de CVs
        print("🧪 Test 1: Búsqueda de CVs")
        response1 = await agent.process_message(
            "Busca perfiles de desarrolladores Python con experiencia en Django",
            test_session
        )
        print(f"Respuesta: {response1}\n")
        
        # Test de selección
        print("🧪 Test 2: Selección de perfil")
        response2 = await agent.process_message(
            "Selecciona el primer perfil",
            test_session
        )
        print(f"Respuesta: {response2}\n")
        
        # Test de búsqueda de credenciales
        print("🧪 Test 3: Búsqueda de credenciales")
        response3 = await agent.process_message(
            "Busca credenciales de AWS",
            test_session
        )
        print(f"Respuesta: {response3}\n")
        
        # Test de exportación
        print("🧪 Test 4: Exportación")
        response4 = await agent.process_message(
            "Quiero exportar los datos seleccionados",
            test_session
        )
        print(f"Respuesta: {response4}\n")
        
        # Verificar estado
        state = agent.get_current_state(test_session)
        print(f"📊 Estado final: {state}")
    
    @staticmethod
    def visualize_graph():
        """Visualiza el grafo del agente (requiere graphviz)"""
        try:
            from IPython.display import Image, display
            graph = build_agent_graph()
            display(Image(graph.get_graph().draw_mermaid_png()))
        except ImportError:
            print("Instala IPython y graphviz para visualizar el grafo")
            graph = build_agent_graph()
            print(graph.get_graph().draw_mermaid())


# ============= Punto de entrada para testing =============
if __name__ == "__main__":
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando sistema de agentes LangGraph")
    
    # Ejecutar tests
    print("=" * 50)
    print("SISTEMA DE AGENTES LANGGRAPH")
    print("=" * 50)
    
    # Visualizar grafo
    print("\n📊 Estructura del Grafo:")
    tester = AgentTester()
    tester.visualize_graph()
    
    # Ejecutar test de flujo
    print("\n🚀 Ejecutando tests de flujo:")
    asyncio.run(tester.test_flow())
    
    print("\n✅ Sistema listo para usar en Streamlit")