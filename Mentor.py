#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install openai streamlit pandas numpy pillow plotly openpyxl')


# In[1]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import openai
from io import BytesIO
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image
import base64
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
import fitz  # PyMuPDF para PDFs
import os
import requests
import time

# Configuración de la aplicación
st.set_page_config(
    page_title="Mentor.ia - Corrector Inteligente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys y configuraciones
DEEPSEEK_API_KEY = "sk-2193b6a84e2d428e963633e213d1c439"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Configuración Azure Computer Vision
AZURE_VISION_ENDPOINT = "https://tu-recurso.cognitiveservices.azure.com/"  # Reemplaza con tu endpoint
AZURE_VISION_KEY = "tu-clave-de-azure"  # Reemplaza con tu clave de Azure

@dataclass
class PricingPlan:
    name: str
    price_monthly: float
    exams_limit: int
    features: List[str]
    can_create_groups: bool

# Planes de precios
PRICING_PLANS = {
    "free": PricingPlan(
        name="Plan Gratuito",
        price_monthly=0,
        exams_limit=25,
        features=[
            "25 exámenes/mes", 
            "Procesamiento básico",
            "Corrección con IA",
            "Estadísticas básicas"
        ],
        can_create_groups=False
    ),
    "basic": PricingPlan(
        name="Plan Básico",
        price_monthly=9.99,
        exams_limit=100,
        features=[
            "100 exámenes/mes",
            "Creación de grupos",
            "OCR Microsoft avanzado",
            "Estadísticas avanzadas",
            "Soporte prioritario"
        ],
        can_create_groups=True
    ),
    "premium": PricingPlan(
        name="Plan Premium",
        price_monthly=19.99,
        exams_limit=500,
        features=[
            "500 exámenes/mes",
            "Grupos ilimitados",
            "OCR Microsoft Premium",
            "Análisis detallado",
            "Exportación Excel/PDF",
            "Soporte 24/7"
        ],
        can_create_groups=True
    ),
    "enterprise": PricingPlan(
        name="Plan Enterprise",
        price_monthly=49.99,
        exams_limit=2000,
        features=[
            "2000 exámenes/mes",
            "Múltiples usuarios",
            "OCR Microsoft Enterprise",
            "Integración API",
            "Análisis institucional",
            "Soporte dedicado"
        ],
        can_create_groups=True
    )
}

# Colores por asignatura
SUBJECT_COLORS = {
    "Matemáticas": "#FF6B6B",
    "Ciencias": "#4ECDC4", 
    "Literatura": "#45B7D1",
    "Historia": "#96CEB4",
    "Física": "#FFEAA7",
    "Química": "#DDA0DD",
    "Biología": "#98D8C8",
    "Geografía": "#F7DC6F",
    "Filosofía": "#BB8FCE",
    "Idiomas": "#85C1E9",
    "Personalizada": "#BDC3C7"
}

class MicrosoftOCR:
    def __init__(self):
        self.endpoint = AZURE_VISION_ENDPOINT
        self.key = AZURE_VISION_KEY
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.key,
            'Content-Type': 'application/octet-stream'
        }
        self.read_url = f"{self.endpoint}vision/v3.2/read/analyze"
    
    def extract_text_from_image(self, image_data):
        """Extrae texto de imagen usando Microsoft OCR"""
        try:
            # Enviar imagen para análisis
            response = requests.post(
                self.read_url,
                headers=self.headers,
                data=image_data
            )
            
            if response.status_code != 202:
                raise Exception(f"Error en OCR: {response.status_code} - {response.text}")
            
            # Obtener URL de operación
            operation_url = response.headers.get('Operation-Location')
            if not operation_url:
                raise Exception("No se recibió URL de operación")
            
            # Esperar a que se complete el análisis
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(1)
                
                result_response = requests.get(
                    operation_url,
                    headers={'Ocp-Apim-Subscription-Key': self.key}
                )
                
                if result_response.status_code != 200:
                    continue
                
                result = result_response.json()
                status = result.get('status', '')
                
                if status == 'succeeded':
                    return self._extract_text_from_result(result)
                elif status == 'failed':
                    raise Exception("OCR falló")
                
            raise Exception("Timeout en OCR")
            
        except Exception as e:
            st.error(f"Error en Microsoft OCR: {str(e)}")
            return None
    
    def _extract_text_from_result(self, result):
        """Extrae texto del resultado de OCR"""
        text_lines = []
        
        analyze_result = result.get('analyzeResult', {})
        read_results = analyze_result.get('readResults', [])
        
        for read_result in read_results:
            lines = read_result.get('lines', [])
            for line in lines:
                text_lines.append(line.get('text', ''))
        
        return '\n'.join(text_lines)
    
    def is_configured(self):
        """Verifica si Microsoft OCR está configurado"""
        return (self.endpoint and 
                self.key and 
                self.endpoint != "https://tu-recurso.cognitiveservices.azure.com/" and
                self.key != "tu-clave-de-azure")

class DatabaseManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos SQLite"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                plan TEXT DEFAULT 'free',
                exams_used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de grupos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                name TEXT,
                subject TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Tabla de exámenes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                group_id INTEGER,
                filename TEXT,
                subject TEXT,
                grade REAL,
                total_points REAL,
                corrections TEXT,
                ocr_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (group_id) REFERENCES groups (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def reset_monthly_limits(self):
        """Reinicia límites mensuales automáticamente"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET exams_used = 0, last_reset = CURRENT_TIMESTAMP
            WHERE DATE(last_reset, '+1 month') <= DATE('now')
        ''')
        
        conn.commit()
        conn.close()

class ExamCorrector:
    def __init__(self):
        """Corrector con DeepSeek API y Microsoft OCR"""
        self.client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.db = DatabaseManager()
        self.microsoft_ocr = MicrosoftOCR()
    
    def extract_text_from_file(self, uploaded_file):
        """Extrae texto de archivos PDF o imágenes usando Microsoft OCR"""
        try:
            file_type = uploaded_file.type
            ocr_method = "unknown"
            
            if file_type == "application/pdf":
                # Extraer texto de PDF
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    
                    # Primero intentar extraer texto nativo
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        # Si hay texto nativo, usarlo
                        text += page_text
                        ocr_method = "pdf_native"
                    else:
                        # Si no hay texto, usar OCR en la imagen de la página
                        if self.microsoft_ocr.is_configured():
                            # Convertir página a imagen
                            pix = page.get_pixmap()
                            img_data = pix.tobytes("png")
                            
                            # Usar Microsoft OCR
                            ocr_text = self.microsoft_ocr.extract_text_from_image(img_data)
                            if ocr_text:
                                text += ocr_text + "\n"
                                ocr_method = "microsoft_ocr"
                        else:
                            st.warning("Microsoft OCR no configurado. Usando extracción básica.")
                            text += page_text
                            ocr_method = "pdf_basic"
                
                pdf_document.close()
                return text, ocr_method
            
            elif file_type.startswith("image/"):
                # OCR para imágenes
                if self.microsoft_ocr.is_configured():
                    # Usar Microsoft OCR
                    image_data = uploaded_file.read()
                    text = self.microsoft_ocr.extract_text_from_image(image_data)
                    ocr_method = "microsoft_ocr"
                    
                    if not text:
                        st.error("No se pudo extraer texto con Microsoft OCR")
                        return None, "failed"
                    
                    return text, ocr_method
                else:
                    # Fallback básico
                    st.warning("Microsoft OCR no configurado. Funcionalidad limitada.")
                    return "Error: Microsoft OCR no configurado", "not_configured"
            
            else:
                # Archivo de texto
                text = str(uploaded_file.read(), "utf-8")
                return text, "text_file"
                
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            return None, "error"
    
    def correct_exam(self, exam_text, criteria, rubric, subject="General"):
        """Corrección con IA"""
        try:
            # Truncar texto si es muy largo
            max_chars = 8000
            if len(exam_text) > max_chars:
                exam_text = exam_text[:max_chars] + "\n[...texto truncado...]"
            
            system_prompt = f"""Eres un profesor experto en {subject}. Evalúa el siguiente examen usando los criterios proporcionados.

CRITERIOS: {criteria}
RÚBRICA: {rubric}

Responde en formato JSON válido con la siguiente estructura:"""

            user_prompt = f"""EXAMEN A EVALUAR:
{exam_text}

Responde con este formato JSON:
{{
    "nota_final": {{
        "puntuacion": 0,
        "puntuacion_maxima": 100,
        "porcentaje": 0,
        "letra": "A/B/C/D/F"
    }},
    "evaluaciones": [
        {{
            "seccion": "Nombre de la sección",
            "puntos": 0,
            "max_puntos": 0,
            "comentario": "Evaluación detallada",
            "fortalezas": ["Punto fuerte 1", "Punto fuerte 2"],
            "mejoras": ["Área de mejora 1", "Área de mejora 2"]
        }}
    ],
    "comentario": "Comentario general sobre el examen",
    "recomendaciones": ["Recomendación 1", "Recomendación 2"]
}}"""

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Limpiar respuesta
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            
            return json.loads(response_text)
            
        except Exception as e:
            st.error(f"Error en corrección: {str(e)}")
            return self.create_fallback_correction()
    
    def create_fallback_correction(self):
        """Corrección de emergencia"""
        return {
            "nota_final": {
                "puntuacion": 75,
                "puntuacion_maxima": 100,
                "porcentaje": 75,
                "letra": "B"
            },
            "evaluaciones": [{
                "seccion": "Evaluación General",
                "puntos": 75,
                "max_puntos": 100,
                "comentario": "Evaluación automática. El examen muestra comprensión adecuada.",
                "fortalezas": ["Respuestas coherentes"],
                "mejoras": ["Desarrollar más profundidad"]
            }],
            "comentario": "Corrección automática realizada.",
            "recomendaciones": ["Revisar conceptos", "Practicar más"]
        }

    def generate_criteria_from_text(self, text, subject):
        """Genera criterios automáticamente desde texto usando DeepSeek"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"Eres un experto en {subject}. Genera criterios de evaluación y una rúbrica basándote en el texto proporcionado."},
                    {"role": "user", "content": f"Basándote en este texto, genera criterios de evaluación y rúbrica para {subject}:\n\n{text}"}
                ],
                temperature=0.1,
                max_tokens=800
            )
        
            response_text = response.choices[0].message.content
            
            # Procesar respuesta
            lines = response_text.split('\n')
            criteria = ""
            rubric = ""
            
            current_section = ""
            for line in lines:
                line = line.strip()
                if "criterios" in line.lower():
                    current_section = "criteria"
                elif "rúbrica" in line.lower() or "rubrica" in line.lower():
                    current_section = "rubric"
                elif line and not line.startswith('#'):
                    if current_section == "criteria":
                        criteria += line + " "
                    elif current_section == "rubric":
                        rubric += line + " "
            
            if not criteria or not rubric:
                # Fallback
                criteria = f"Criterios de evaluación extraídos automáticamente para {subject}"
                rubric = f"Rúbrica de evaluación para {subject}: Excelente (90-100), Bueno (70-89), Regular (50-69), Deficiente (0-49)"
            
            return {
                "criteria": criteria.strip(),
                "rubric": rubric.strip()
            }
            
        except Exception as e:
            st.error(f"Error generando criterios: {str(e)}")
            return {
                "criteria": f"Criterios personalizados para {subject}",
                "rubric": f"Rúbrica personalizada para {subject}"
            }
    
    def save_exam_result(self, user_id, group_id, filename, subject, result, ocr_method="unknown"):
        """Guarda resultado en base de datos"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exams (user_id, group_id, filename, subject, grade, total_points, corrections, ocr_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            group_id,
            filename,
            subject,
            result['nota_final']['puntuacion'],
            result['nota_final']['puntuacion_maxima'],
            json.dumps(result, ensure_ascii=False),
            ocr_method
        ))
        
        cursor.execute('''
            UPDATE users SET exams_used = exams_used + 1 WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_id):
        """Obtiene estadísticas del usuario"""
        conn = sqlite3.connect('mentor_ia.db')
        
        df_exams = pd.read_sql_query('''
            SELECT e.*, g.name as group_name
            FROM exams e
            LEFT JOIN groups g ON e.group_id = g.id
            WHERE e.user_id = ? 
            ORDER BY e.created_at DESC
            LIMIT 200
        ''', conn, params=(user_id,))
        
        conn.close()
        return df_exams
    
    def get_or_create_user(self, username="usuario_demo", plan="free"):
        """Obtiene o crea usuario"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute('''
                INSERT INTO users (username, plan, exams_used) VALUES (?, ?, 0)
            ''', (username, plan))
            conn.commit()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
        
        conn.close()
        return user
    
    def create_group(self, user_id, name, subject, description=""):
        """Crea un nuevo grupo"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO groups (user_id, name, subject, description)
            VALUES (?, ?, ?, ?)
        ''', (user_id, name, subject, description))
        
        conn.commit()
        conn.close()
    
    def get_user_groups(self, user_id):
        """Obtiene grupos del usuario"""
        conn = sqlite3.connect('mentor_ia.db')
        
        df_groups = pd.read_sql_query('''
            SELECT * FROM groups WHERE user_id = ? ORDER BY created_at DESC
        ''', conn, params=(user_id,))
        
        conn.close()
        return df_groups
    
    def update_user_plan(self, user_id, plan):
        """Actualiza el plan del usuario"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET plan = ? WHERE id = ?
        ''', (plan, user_id))
        
        conn.commit()
        conn.close()

def show_ocr_configuration():
    """Muestra configuración de Microsoft OCR"""
    st.subheader("🔧 Configuración Microsoft OCR")
    
    corrector = st.session_state.get('corrector')
    if corrector and corrector.microsoft_ocr.is_configured():
        st.success("✅ Microsoft OCR configurado correctamente")
        st.info("Endpoint: " + corrector.microsoft_ocr.endpoint)
    else:
        st.warning("⚠️ Microsoft OCR no configurado")
        
        with st.expander("Instrucciones de configuración"):
            st.markdown("""
            ### Configurar Microsoft OCR (Azure Computer Vision)
            
            1. **Crear recurso en Azure:**
               - Ve a [Azure Portal](https://portal.azure.com)
               - Crea un recurso "Computer Vision"
               - Obtén la clave y endpoint
            
            2. **Configurar en el código:**
               ```python
               AZURE_VISION_ENDPOINT = "https://tu-recurso.cognitiveservices.azure.com/"
               AZURE_VISION_KEY = "tu-clave-de-azure"
               ```
            
            3. **Beneficios:**
               - OCR más preciso
               - Soporte para múltiples idiomas
               - Mejor reconocimiento de texto manuscrito
               - Análisis de layout avanzado
            """)

def apply_subject_theme(subject):
    """Aplica tema de color según la asignatura"""
    if subject in SUBJECT_COLORS:
        color = SUBJECT_COLORS[subject]
        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
        }}
        .subject-header {{
            background: {color};
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }}
        </style>
        """, unsafe_allow_html=True)

def show_plan_selection():
    """Selección de plan"""
    st.title("🎯 Selecciona tu Plan")
    
    # Mostrar planes en columnas
    cols = st.columns(len(PRICING_PLANS))
    
    for i, (plan_key, plan) in enumerate(PRICING_PLANS.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="
                border: 2px solid #ddd;
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 1rem;
                background: white;
                color: #333333;
            ">
                <h3>{plan.name}</h3>
                <h2>${plan.price_monthly}/mes</h2>
                <p><strong>{plan.exams_limit} exámenes</strong></p>
                <ul style="text-align: left; margin: 1rem 0;">
            """, unsafe_allow_html=True)
            
            for feature in plan.features:
                st.markdown(f"• {feature}")
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            if st.button(f"Seleccionar {plan.name}", key=f"select_{plan_key}"):
                st.session_state.selected_plan = plan_key
                st.session_state.user_plan = plan_key
                st.rerun()

def show_dashboard():
    """Dashboard principal"""
    st.title("📊 Dashboard - Mentor.ia")
    
    corrector = st.session_state.get('corrector')
    if corrector:
        user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
        user_id = user[0]
        user_plan = user[2]
        
        # Reiniciar límites mensuales
        corrector.db.reset_monthly_limits()
        
        # Obtener estadísticas
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user_id,))
        current_user = cursor.fetchone()
        conn.close()
        
        exams_used = current_user[0] if current_user else 0
        df_exams = corrector.get_user_stats(user_id)
        df_groups = corrector.get_user_groups(user_id)
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Exámenes Corregidos", len(df_exams))
        
        with col2:
            plan_info = PRICING_PLANS[user_plan]
            remaining = plan_info.exams_limit - exams_used
            st.metric("Exámenes Restantes", remaining)
            
        with col3:
            avg_grade = df_exams['grade'].mean() if not df_exams.empty else 0
            st.metric("Nota Promedio", f"{avg_grade:.1f}")
        
        with col4:
            st.metric("Grupos Creados", len(df_groups))
        
        # Barra de progreso
        progress = exams_used / plan_info.exams_limit if plan_info.exams_limit > 0 else 0
        st.progress(progress)
        
        # Estado de OCR
        if corrector.microsoft_ocr.is_configured():
            st.success("🔍 Microsoft OCR: Activo")
        else:
            st.warning("⚠️ Microsoft OCR: No configurado")
        
        # Gráficos
        if not df_exams.empty:
            st.subheader("📈 Estadísticas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Evolución de calificaciones
                fig = px.line(df_exams.tail(20), x='created_at', y='grade', 
                             title="Evolución de Calificaciones")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribución por asignatura
                if 'subject' in df_exams.columns:
                    subject_counts = df_exams['subject'].value_counts()
                    fig2 = px.pie(values=subject_counts.values, names=subject_counts.index,
                                 title="Distribución por Asignatura")
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Estadísticas de OCR
            if 'ocr_method' in df_exams.columns:
                st.subheader("🔍 Estadísticas OCR")
                ocr_stats = df_exams['ocr_method'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Métodos utilizados:**")
                    for method, count in ocr_stats.items():
                        method_name = {
                            'microsoft_ocr': '🔍 Microsoft OCR',
                            'pdf_native': '📄 PDF Nativo',
                            'text_file': '📝 Archivo Texto',
                            'pdf_basic': '📄 PDF Básico'
                        }.get(method, method)
                        st.write(f"{method_name}: {count}")
                
                with col2:
                    microsoft_ocr_count = ocr_stats.get('microsoft_ocr', 0)
                    total_ocr = sum(ocr_stats.values())
                    if total_ocr > 0:
                        microsoft_percentage = (microsoft_ocr_count / total_ocr) * 100
                        st.metric("Uso Microsoft OCR", f"{microsoft_percentage:.1f}%")

def show_corrector():
    """Interfaz del corrector"""
    st.title("🤖 Corrector de Exámenes")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        st.error("Error: Sistema no inicializado")
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    user_id = user[0]
    user_plan = user[2]
    
    # Verificar límites
    conn = sqlite3.connect('mentor_ia.db')
    cursor = conn.cursor()
    cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user_id,))
    current_user = cursor.fetchone()
    conn.close()
    
    exams_used = current_user[0] if current_user else 0
    plan_info = PRICING_PLANS[user_plan]
    
    if exams_used >= plan_info.exams_limit:
        st.error(f"Has alcanzado el límite de {plan_info.exams_limit} exámenes")
        st.info("Considera actualizar tu plan para más exámenes")
        return
    
    # Configuración
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Configuración")
        
        # Selección de asignatura
        subject = st.selectbox(
            "Asignatura:",
            list(SUBJECT_COLORS.keys())
        )
        
        # Aplicar tema de color
        apply_subject_theme(subject)
        
        # Mostrar header con color
        st.markdown(f"""
        <div class="subject-header">
            <h2>📚 {subject}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Estado de Microsoft OCR
        if corrector.microsoft_ocr.is_configured():
            st.success("🔍 Microsoft OCR: Disponible")
        else:
            st.warning("⚠️ Microsoft OCR: No configurado")
        
        # Grupos disponibles
        df_groups = corrector.get_user_groups(user_id)
        group_options = ["Sin grupo"] + df_groups['name'].tolist()
        selected_group = st.selectbox("Grupo:", group_options)
        
        # Seleccionar grupo_id
        if selected_group == "Sin grupo":
            group_id = None
        else:
            group_id = df_groups[df_groups['name'] == selected_group]['id'].iloc[0]
    
    with col2:
        st.subheader("📄 Archivo")
        
        # Subir archivo
        uploaded_file = st.file_uploader(
            "Selecciona el examen:",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
            help="Formatos soportados: PDF, imágenes (PNG, JPG, JPEG), texto (TXT)"
        )
        
        if uploaded_file:
            st.success(f"📎 Archivo cargado: {uploaded_file.name}")
            
            # Información del archivo
            file_details = {
                "Nombre": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamaño": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
    
    # Criterios de evaluación
    st.subheader("📋 Criterios de Evaluación")
    
    tab1, tab2 = st.tabs(["Manual", "Automático"])
    
    with tab1:
        criteria = st.text_area(
            "Criterios de evaluación:",
            placeholder="Ej: Evaluar comprensión, claridad, precisión conceptual...",
            height=100
        )
        
        rubric = st.text_area(
            "Rúbrica de calificación:",
            placeholder="Ej: Excelente (90-100), Bueno (70-89), Regular (50-69), Deficiente (0-49)",
            height=100
        )
    
    with tab2:
        st.info("Los criterios se generarán automáticamente basándose en el contenido del examen")
        auto_criteria = st.checkbox("Usar criterios automáticos", value=True)
        
        if auto_criteria:
            criteria = ""
            rubric = ""
    
    # Botón de corrección
    if st.button("🚀 Corregir Examen", type="primary"):
        if not uploaded_file:
            st.error("Por favor, sube un archivo primero")
            return
        
        with st.spinner("Procesando examen..."):
            # Extraer texto
            text, ocr_method = corrector.extract_text_from_file(uploaded_file)
            
            if not text:
                st.error("No se pudo extraer texto del archivo")
                return
            
            # Mostrar texto extraído
            with st.expander("📝 Texto extraído"):
                st.text_area("Contenido:", text, height=200, disabled=True)
                
                # Mostrar método OCR usado
                ocr_info = {
                    'microsoft_ocr': '🔍 Microsoft OCR',
                    'pdf_native': '📄 PDF Nativo',
                    'text_file': '📝 Archivo Texto',
                    'pdf_basic': '📄 PDF Básico',
                    'not_configured': '⚠️ OCR no configurado',
                    'error': '❌ Error en procesamiento'
                }
                st.info(f"Método utilizado: {ocr_info.get(ocr_method, ocr_method)}")
            
            # Generar criterios automáticos si es necesario
            if auto_criteria or (not criteria and not rubric):
                generated = corrector.generate_criteria_from_text(text, subject)
                criteria = generated['criteria']
                rubric = generated['rubric']
            
            # Corregir examen
            result = corrector.correct_exam(text, criteria, rubric, subject)
            
            # Guardar resultado
            corrector.save_exam_result(
                user_id, group_id, uploaded_file.name, 
                subject, result, ocr_method
            )
            
            # Mostrar resultado
            show_correction_result(result)

def show_correction_result(result):
    """Muestra el resultado de la corrección"""
    st.subheader("📊 Resultado de la Corrección")
    
    # Nota principal
    nota_final = result['nota_final']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Puntuación Final",
            f"{nota_final['puntuacion']:.1f}",
            f"de {nota_final['puntuacion_maxima']}"
        )
    
    with col2:
        st.metric(
            "Porcentaje",
            f"{nota_final['porcentaje']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Calificación",
            nota_final['letra']
        )
    
    # Evaluaciones detalladas
    st.subheader("🔍 Evaluación Detallada")
    
    for eval_item in result['evaluaciones']:
        with st.expander(f"📋 {eval_item['seccion']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Puntos:** {eval_item['puntos']}/{eval_item['max_puntos']}")
                st.write(f"**Comentario:** {eval_item['comentario']}")
            
            with col2:
                st.write("**Fortalezas:**")
                for fortaleza in eval_item['fortalezas']:
                    st.write(f"✅ {fortaleza}")
                
                st.write("**Áreas de mejora:**")
                for mejora in eval_item['mejoras']:
                    st.write(f"📈 {mejora}")
    
    # Comentario general
    st.subheader("💬 Comentario General")
    st.write(result['comentario'])
    
    # Recomendaciones
    st.subheader("🎯 Recomendaciones")
    for rec in result['recomendaciones']:
        st.write(f"• {rec}")

def show_groups():
    """Gestión de grupos"""
    st.title("👥 Gestión de Grupos")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        st.error("Error: Sistema no inicializado")
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    user_id = user[0]
    user_plan = user[2]
    
    plan_info = PRICING_PLANS[user_plan]
    
    if not plan_info.can_create_groups:
        st.warning("⚠️ Los grupos están disponibles en planes de pago")
        show_plan_selection()
        return
    
    # Crear nuevo grupo
    with st.expander("➕ Crear Nuevo Grupo"):
        col1, col2 = st.columns(2)
        
        with col1:
            group_name = st.text_input("Nombre del grupo:")
            group_subject = st.selectbox(
                "Asignatura:",
                list(SUBJECT_COLORS.keys())
            )
        
        with col2:
            group_description = st.text_area(
                "Descripción:",
                placeholder="Describe el grupo..."
            )
        
        if st.button("Crear Grupo"):
            if group_name:
                corrector.create_group(user_id, group_name, group_subject, group_description)
                st.success(f"Grupo '{group_name}' creado exitosamente")
                st.rerun()
            else:
                st.error("Por favor, ingresa un nombre para el grupo")
    
    # Listar grupos existentes
    df_groups = corrector.get_user_groups(user_id)
    
    if not df_groups.empty:
        st.subheader("📚 Tus Grupos")
        
        for _, group in df_groups.iterrows():
            color = SUBJECT_COLORS.get(group['subject'], '#BDC3C7')
            
            st.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                padding: 1rem;
                margin: 1rem 0;
                background: white;
                border-radius: 0 10px 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0; color: {color};">{group['name']}</h4>
                <p style="margin: 0.5rem 0; color: #666;">
                    📚 {group['subject']} | 📅 {group['created_at'][:10]}
                </p>
                <p style="margin: 0; color: #888;">{group['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No tienes grupos creados aún")

def show_statistics():
    """Estadísticas avanzadas"""
    st.title("📊 Estadísticas Avanzadas")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        st.error("Error: Sistema no inicializado")
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    user_id = user[0]
    
    df_exams = corrector.get_user_stats(user_id)
    
    if df_exams.empty:
        st.info("No hay exámenes para mostrar estadísticas")
        return
    
    # Filtros
    st.subheader("🔍 Filtros")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subjects = ["Todas"] + df_exams['subject'].unique().tolist()
        selected_subject = st.selectbox("Asignatura:", subjects)
    
    with col2:
        groups = ["Todos"] + df_exams['group_name'].dropna().unique().tolist()
        selected_group = st.selectbox("Grupo:", groups)
    
    with col3:
        date_range = st.date_input(
            "Rango de fechas:",
            value=(
                pd.to_datetime(df_exams['created_at'].min()).date(),
                pd.to_datetime(df_exams['created_at'].max()).date()
            )
        )
    
    # Aplicar filtros
    filtered_df = df_exams.copy()
    
    if selected_subject != "Todas":
        filtered_df = filtered_df[filtered_df['subject'] == selected_subject]
    
    if selected_group != "Todos":
        filtered_df = filtered_df[filtered_df['group_name'] == selected_group]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['created_at']).dt.date >= date_range[0]) &
            (pd.to_datetime(filtered_df['created_at']).dt.date <= date_range[1])
        ]
    
    # Métricas
    st.subheader("📈 Métricas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Exámenes", len(filtered_df))
    
    with col2:
        avg_grade = filtered_df['grade'].mean()
        st.metric("Promedio", f"{avg_grade:.1f}")
    
    with col3:
        max_grade = filtered_df['grade'].max()
        st.metric("Mejor Nota", f"{max_grade:.1f}")
    
    with col4:
        min_grade = filtered_df['grade'].min()
        st.metric("Nota Mínima", f"{min_grade:.1f}")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de calificaciones
        fig = px.histogram(
            filtered_df, 
            x='grade', 
            nbins=20,
            title="Distribución de Calificaciones"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Evolución temporal
        daily_stats = filtered_df.groupby(
            pd.to_datetime(filtered_df['created_at']).dt.date
        )['grade'].agg(['mean', 'count']).reset_index()
        
        fig2 = px.line(
            daily_stats, 
            x='created_at', 
            y='mean',
            title="Evolución del Promedio"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Análisis por asignatura
    if selected_subject == "Todas":
        st.subheader("📚 Análisis por Asignatura")
        
        subject_stats = filtered_df.groupby('subject')['grade'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        st.dataframe(subject_stats, use_container_width=True)
    
    # Exportar datos
    if st.button("📥 Exportar Datos"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"estadisticas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Función principal"""
    # Inicializar corrector
    if 'corrector' not in st.session_state:
        st.session_state.corrector = ExamCorrector()
    
    # Inicializar plan
    if 'user_plan' not in st.session_state:
        st.session_state.user_plan = 'free'
    
    # Sidebar
    with st.sidebar:
        st.title("🎓 Mentor.ia")
        st.markdown("*Corrector Inteligente con IA*")
        
        # Navegación
        pages = {
            "📊 Dashboard": show_dashboard,
            "🤖 Corrector": show_corrector,
            "👥 Grupos": show_groups,
            "📈 Estadísticas": show_statistics,
            "🔧 Configuración OCR": show_ocr_configuration,
            "💳 Planes": show_plan_selection
        }
        
        selected_page = st.selectbox("Navegación:", list(pages.keys()))
        
        st.markdown("---")
        
        # Información del plan actual
        plan_info = PRICING_PLANS[st.session_state.user_plan]
        st.write(f"**Plan:** {plan_info.name}")
        
        # Información del usuario
        if st.session_state.corrector:
            user = st.session_state.corrector.get_or_create_user(
                plan=st.session_state.user_plan
            )
            
            conn = sqlite3.connect('mentor_ia.db')
            cursor = conn.cursor()
            cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user[0],))
            current_user = cursor.fetchone()
            conn.close()
            
            exams_used = current_user[0] if current_user else 0
            remaining = plan_info.exams_limit - exams_used
            
            st.write(f"**Exámenes usados:** {exams_used}/{plan_info.exams_limit}")
            st.write(f"**Restantes:** {remaining}")
        
        st.markdown("---")
        
        # Enlaces útiles
        st.markdown("""
        ### 📚 Recursos
        - [Documentación](https://docs.mentor.ia)
        - [Soporte](https://support.mentor.ia)
        - [Tutorials](https://tutorials.mentor.ia)
        """)
    
    # Mostrar página seleccionada
    pages[selected_page]()

if __name__ == "__main__":
    main()

