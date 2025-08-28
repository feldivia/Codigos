#!/usr/bin/env python3
"""
API FastAPI para búsqueda vectorial en LanceDB
Expone el método vector_search como endpoint REST
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import lancedb
import numpy as np
import pandas as pd
import uvicorn
import os
from datetime import datetime


# ===========================
# MODELOS PYDANTIC
# ===========================

class VectorSearchRequest(BaseModel):
    """Modelo de request para búsqueda vectorial"""
    embedding: List[float] = Field(
        ..., 
        description="Vector de embeddings (1536 dimensiones para ada-002)",
        min_items=1536,
        max_items=1536
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Número de resultados a retornar",
        ge=1,
        le=50
    )

class SearchResult(BaseModel):
    """Modelo de un resultado individual"""
    index: int
    score: float
    personal_id: Optional[str]
    nombres: Optional[str]
    apellido_paterno: Optional[str]
    apellido_materno: Optional[str]
    cargo: Optional[str]
    email: Optional[str]
    los: Optional[str]
    sublos: Optional[str]
    antecedente: Optional[str]
    habilidades: Optional[str]
    areas_experiencia: Optional[str]
    industrias: Optional[str]

class VectorSearchResponse(BaseModel):
    """Modelo de response para búsqueda vectorial"""
    success: bool
    total_results: int
    search_results: List[SearchResult]
    execution_time_ms: float
    message: Optional[str] = None

class HealthCheckResponse(BaseModel):
    """Modelo para health check"""
    status: str
    database_connected: bool
    table_name: str
    total_records: int
    timestamp: str


# ===========================
# CONFIGURACIÓN
# ===========================

class Config:
    """Configuración de la API"""
    DB_URI = os.getenv("LANCEDB_URI", "./data/target_db")
    TABLE_NAME = os.getenv("LANCEDB_TABLE", "personal_embeddings")
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"


# ===========================
# SERVICIO DE BÚSQUEDA
# ===========================

class VectorSearchService:
    """Servicio para búsqueda vectorial en LanceDB"""
    
    def __init__(self, db_uri: str, table_name: str):
        """
        Inicializa el servicio de búsqueda
        
        Args:
            db_uri: URI de la base de datos LanceDB
            table_name: Nombre de la tabla
        """
        self.db_uri = db_uri
        self.table_name = table_name
        self.db = None
        self.table = None
        self._connect()
    
    def _connect(self):
        """Conecta a la base de datos LanceDB"""
        try:
            self.db = lancedb.connect(self.db_uri)
            self.table = self.db.open_table(self.table_name)
            print(f"✅ Conectado a LanceDB: {self.db_uri}/{self.table_name}")
        except Exception as e:
            print(f"❌ Error conectando a LanceDB: {str(e)}")
            raise
    
    def vector_search(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda vectorial usando embeddings
        
        Args:
            embedding: Vector de embeddings
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados con scores de similitud
        """
        try:
            # Realizar búsqueda vectorial nativa en LanceDB
            results = (
                self.table.search(embedding)
                .metric("cosine")
                .limit(top_k)
                .to_pandas()
            )
            
            # Procesar resultados
            search_results = []
            for idx, row in results.iterrows():
                # Convertir distancia a score de similitud
                distance = row.get('_distance', 0)
                similarity_score = max(0, 1 - (distance / 2))
                
                # Preparar resultado
                result = {
                    'index': int(idx),
                    'score': float(similarity_score),
                    'personal_id': row.get('personal_id'),
                    'nombres': row.get('nombres'),
                    'apellido_paterno': row.get('apellido_paterno'),
                    'apellido_materno': row.get('apellido_materno'),
                    'cargo': row.get('cargo'),
                    'email': row.get('email'),
                    'los': row.get('los'),
                    'sublos': row.get('sublos'),
                    'antecedente': row.get('antecedente'),
                    'habilidades': row.get('habilidades'),
                    'areas_experiencia': row.get('areas_experiencia'),
                    'industrias': row.get('industrias')
                }
                
                # Limpiar valores None y NaN
                result = {k: (v if pd.notna(v) else None) for k, v in result.items()}
                
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error en búsqueda vectorial: {str(e)}")
            raise
    
    def get_table_info(self) -> Dict[str, Any]:
        """Obtiene información de la tabla"""
        try:
            df = self.table.to_pandas()
            return {
                'total_records': len(df),
                'columns': list(df.columns),
                'database_uri': self.db_uri,
                'table_name': self.table_name
            }
        except Exception as e:
            return {'error': str(e)}


# ===========================
# INICIALIZACIÓN DE LA API
# ===========================

# Crear instancia de FastAPI
app = FastAPI(
    title="CV Vector Search API",
    description="API para búsqueda vectorial de CVs usando embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicio
search_service = None


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    global search_service
    try:
        search_service = VectorSearchService(
            db_uri=Config.DB_URI,
            table_name=Config.TABLE_NAME
        )
        print("🚀 API iniciada correctamente")
    except Exception as e:
        print(f"❌ Error al iniciar: {str(e)}")
        raise


# ===========================
# ENDPOINTS
# ===========================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz"""
    return {
        "message": "CV Vector Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health_check():
    """
    Health check del servicio
    Verifica la conexión a la base de datos
    """
    try:
        info = search_service.get_table_info()
        return HealthCheckResponse(
            status="healthy",
            database_connected=True,
            table_name=Config.TABLE_NAME,
            total_records=info.get('total_records', 0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/vector_search", response_model=VectorSearchResponse, tags=["Search"])
async def vector_search(request: VectorSearchRequest):
    """
    Realiza búsqueda vectorial usando embeddings
    
    Recibe un vector de embeddings y retorna los top_k resultados más similares.
    
    - **embedding**: Vector de 1536 dimensiones (para ada-002)
    - **top_k**: Número de resultados a retornar (1-50)
    
    Returns:
        Lista de candidatos ordenados por similitud
    """
    try:
        # Medir tiempo de ejecución
        import time
        start_time = time.time()
        
        # Validar dimensiones del embedding
        if len(request.embedding) != 1536:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Embedding debe tener 1536 dimensiones, recibido: {len(request.embedding)}"
            )
        
        # Realizar búsqueda
        search_results = search_service.vector_search(
            embedding=request.embedding,
            top_k=request.top_k
        )
        
        # Calcular tiempo de ejecución
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Preparar respuesta
        return VectorSearchResponse(
            success=True,
            total_results=len(search_results),
            search_results=search_results,
            execution_time_ms=round(execution_time_ms, 2),
            message=f"Búsqueda completada exitosamente"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en búsqueda vectorial: {str(e)}"
        )


@app.post("/vector_search_batch", tags=["Search"])
async def vector_search_batch(embeddings: List[List[float]], top_k: int = 5):
    """
    Realiza múltiples búsquedas vectoriales en batch
    
    Args:
        embeddings: Lista de vectores de embeddings
        top_k: Número de resultados por búsqueda
    
    Returns:
        Lista de resultados para cada embedding
    """
    try:
        batch_results = []
        
        for embedding in embeddings:
            if len(embedding) != 1536:
                batch_results.append({
                    "error": f"Dimensión incorrecta: {len(embedding)}"
                })
                continue
            
            search_results = search_service.vector_search(
                embedding=embedding,
                top_k=top_k
            )
            batch_results.append({
                "results": search_results
            })
        
        return {
            "success": True,
            "total_searches": len(embeddings),
            "batch_results": batch_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en búsqueda batch: {str(e)}"
        )


# ===========================
# PUNTO DE ENTRADA
# ===========================

if __name__ == "__main__":
    print("🚀 Iniciando API de Búsqueda Vectorial...")
    print(f"📍 URL: http://{Config.HOST}:{Config.PORT}")
    print(f"📚 Documentación: http://{Config.HOST}:{Config.PORT}/docs")
    print(f"🗄️ Base de datos: {Config.DB_URI}")
    print(f"📊 Tabla: {Config.TABLE_NAME}")
    
    uvicorn.run(
        "fastapi_vector_search:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD
    )