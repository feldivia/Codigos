"""
Solución 3: Extracción de tablas usando Docling
Docling es una biblioteca moderna de IBM que usa AI para detectar tablas
Ventajas: Mejor detección de estructura, maneja tablas complejas, preserva formato
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys
import io
import re
import os

# Configurar encoding para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# IMPORTANTE: Deshabilitar symlinks en Windows para evitar errores de permisos
# Esto DEBE configurarse ANTES de importar cualquier librería de Hugging Face
if sys.platform == 'win32':
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Importar Docling
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️  Docling no está instalado. Ejecuta: pip install docling")
    print("   Para instalación completa con OCR: pip install docling[ocr]")


class DoclingFinancialExtractor:
    """Extractor especializado usando Docling (AI-powered) para estados financieros"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.debug_mode = True

        if not DOCLING_AVAILABLE:
            raise ImportError("Docling no está instalado. Ejecuta: pip install docling")

    def extract_all(self, pages='all') -> List[Dict]:
        """Extracción con Docling (AI-powered)"""
        results = []

        print("Extrayendo tablas con Docling (AI-powered)...")
        print("  (Esto puede tomar unos momentos mientras los modelos AI analizan el PDF)")

        try:
            # Configurar opciones para mejorar la extracción
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Cambiar a True si el PDF es escaneado
            pipeline_options.do_table_structure = True  # Detectar estructura de tablas

            # Crear conversor con opciones de formato
            from docling.document_converter import PdfFormatOption

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            # Convertir documento
            if self.debug_mode:
                print(f"  Procesando: {self.pdf_path}")

            result = converter.convert(self.pdf_path)

            # Acceder al documento convertido
            doc = result.document

            if self.debug_mode:
                # Contar tablas
                tables_list = list(doc.tables) if hasattr(doc, 'tables') else []
                total_tables = len(tables_list)
                print(f"  ✓ Docling detectó {total_tables} tablas")

            # Extraer tablas
            if hasattr(doc, 'tables'):
                for table_idx, table in enumerate(doc.tables):
                    try:
                        # Convertir a DataFrame
                        df = table.export_to_dataframe()

                        if df is not None and not df.empty:
                            # Limpiar DataFrame
                            df_clean = self._clean_dataframe(df)

                            if df_clean is not None and len(df_clean) >= 2:
                                # Intentar obtener número de página
                                # Docling puede proporcionar prov (provenance) con información de página
                                page_num = table_idx + 1  # Default
                                if hasattr(table, 'prov') and hasattr(table.prov, 'page'):
                                    page_num = table.prov.page
                                elif hasattr(table, 'page'):
                                    page_num = table.page

                                results.append({
                                    'page': page_num,
                                    'data': df_clean,
                                    'method': 'docling_ai',
                                    'confidence': self._calculate_confidence(df_clean),
                                    'table_metadata': {
                                        'num_rows': len(df_clean),
                                        'num_cols': len(df_clean.columns)
                                    }
                                })

                                if self.debug_mode:
                                    print(f"    ✓ Tabla {table_idx + 1}: {len(df_clean)} filas x {len(df_clean.columns)} cols")

                    except Exception as e:
                        if self.debug_mode:
                            print(f"    ✗ Error procesando tabla {table_idx + 1}: {str(e)[:50]}")
                        continue
            else:
                print("  ⚠️  El documento no contiene tablas o no se pudieron detectar")

            print(f"  [OK] Docling: {len(results)} tablas extraídas")

        except Exception as e:
            print(f"  [ERROR] Docling: {str(e)}")
            import traceback
            if self.debug_mode:
                traceback.print_exc()

        return results

    def _clean_dataframe(self, df):
        """Limpia y normaliza DataFrame PRESERVANDO INDENTACIÓN"""
        if df.empty or len(df) < 2:
            return None

        # Limpiar todas las celdas
        for col_idx, col in enumerate(df.columns):
            if col_idx == 0:
                # Primera columna: preservar indentación
                df[col] = df[col].apply(lambda x: self._clean_cell_preserve_indent(str(x)) if pd.notna(x) else '')
            else:
                # Otras columnas: limpiar completamente
                df[col] = df[col].apply(lambda x: self._clean_cell_full(str(x)) if pd.notna(x) else '')

        # Eliminar filas completamente vacías
        df = df.dropna(how='all')

        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how='all')

        # Eliminar columnas que solo contienen espacios
        for col in df.columns:
            if df[col].apply(lambda x: not str(x).strip()).all():
                df = df.drop(columns=[col])

        if df.empty or len(df) < 2:
            return None

        # Resetear índices
        df = df.reset_index(drop=True)

        # Detectar y usar header si existe
        if self._is_header_row(df.iloc[0]):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)

        return df

    def _clean_cell_preserve_indent(self, cell_value):
        """Limpia una celda preservando la indentación inicial"""
        # Convertir a string y manejar None/NaN
        if pd.isna(cell_value) or cell_value is None:
            return ''

        cell_str = str(cell_value)

        # 1. Eliminar tags HTML
        cleaned = re.sub(r'<[^>]+>', '', cell_str)

        # 2. Reemplazar TODOS los tipos de saltos de línea por UN SOLO espacio
        cleaned = re.sub(r'[\r\n]+', ' ', cleaned)

        # 3. Eliminar múltiples espacios consecutivos
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        # 4. Reemplazar /dollarsign por $
        cleaned = cleaned.replace('/dollarsign', '$')

        # 5. Solo eliminar espacios finales, NO iniciales (preservar indentación)
        cleaned = cleaned.rstrip()

        return cleaned

    def _clean_cell_full(self, cell_value):
        """Limpia una celda completamente (sin preservar indentación)"""
        # Convertir a string y manejar None/NaN
        if pd.isna(cell_value) or cell_value is None:
            return ''

        cell_str = str(cell_value)

        # 1. Eliminar tags HTML
        cleaned = re.sub(r'<[^>]+>', '', cell_str)

        # 2. Reemplazar TODOS los tipos de saltos de línea
        cleaned = re.sub(r'[\r\n]+', ' ', cleaned)

        # 3. Eliminar múltiples espacios consecutivos
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        # 4. Reemplazar /dollarsign por $
        cleaned = cleaned.replace('/dollarsign', '$')

        # 5. Eliminar espacios extras
        cleaned = cleaned.strip()

        return cleaned

    def _is_header_row(self, row):
        """Detecta si una fila es un header"""
        non_numeric = 0
        total_non_empty = 0

        for val in row:
            if pd.notna(val) and str(val).strip():
                total_non_empty += 1
                val_str = str(val).replace(',', '').replace('.', '')
                if not val_str.replace('-', '').isdigit():
                    non_numeric += 1

        return total_non_empty > 0 and (non_numeric / total_non_empty) > 0.6

    def _calculate_confidence(self, df):
        """Calcula score de confianza para la tabla extraída"""
        if df.empty:
            return 0

        score = 70  # Base score (Docling tiene buena calidad)

        # Bonificar por tamaño razonable
        if len(df) >= 5:
            score += 10
        if len(df.columns) >= 3:
            score += 10

        # Bonificar por contenido numérico (típico de estados financieros)
        numeric_cells = 0
        total_cells = 0

        for col in df.columns:
            for val in df[col]:
                if pd.notna(val) and str(val).strip():
                    total_cells += 1
                    val_str = str(val).replace(',', '').replace('.', '').replace('-', '').replace('(', '').replace(')', '')
                    if val_str.isdigit():
                        numeric_cells += 1

        if total_cells > 0:
            numeric_ratio = numeric_cells / total_cells
            if numeric_ratio > 0.4:
                score += 10

        return min(score, 100)

    def export_results(self, results: List[Dict], output_path: str):
        """Exporta TODAS las tablas encontradas a Excel"""
        if not results:
            print("No hay resultados para exportar")
            return

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Contador de tablas por página
            page_table_counters = {}

            for result in results:
                page = result['page']

                # Incrementar contador
                if page not in page_table_counters:
                    page_table_counters[page] = 1
                else:
                    page_table_counters[page] += 1

                table_num = page_table_counters[page]

                # Nombre de hoja
                sheet_name = f"P{page}_T{table_num}"
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]

                # Metadatos
                metadata = pd.DataFrame([{
                    'Página PDF': page,
                    'Tabla': table_num,
                    'Método': result['method'],
                    'Confianza': f"{result['confidence']:.1f}%",
                    'Filas': len(result['data']),
                    'Columnas': len(result['data'].columns)
                }])

                # Escribir metadatos y datos
                metadata.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
                result['data'].to_excel(writer, sheet_name=sheet_name, startrow=3, index=False)

        print(f"\n[OK] Exportado: {output_path}")
        print(f"  Total de tablas exportadas: {len(results)}")
        print(f"  Páginas con tablas: {sorted(set(r['page'] for r in results))}")


# USO
if __name__ == "__main__":
    if not DOCLING_AVAILABLE:
        print("\n❌ ERROR: Docling no está instalado")
        print("\nPara instalar Docling:")
        print("  pip install docling")
        print("\nPara instalación completa con OCR (PDFs escaneados):")
        print("  pip install docling[ocr]")
        sys.exit(1)

    try:
        extractor = DoclingFinancialExtractor('Estados_Financieros_COPEC.pdf')
        results = extractor.extract_all(pages='all')
        extractor.export_results(results, 'tablas_docling_COPEC.xlsx')

        print(f"\n{'='*60}")
        print(f"RESUMEN")
        print(f"{'='*60}")
        print(f"Total: {len(results)} tablas")
        print(f"Método: Docling AI-powered")

        # Resumen por página
        by_page = {}
        for r in results:
            page = r['page']
            if page not in by_page:
                by_page[page] = 0
            by_page[page] += 1

        print(f"\nTablas por página:")
        for page in sorted(by_page.keys()):
            print(f"  Página {page}: {by_page[page]} tabla(s)")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
