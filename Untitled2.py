#!/usr/bin/env python
# coding: utf-8

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
import cv2

# Configuración de la aplicación
st.set_page_config(
    page_title="Mentor.ia - Corrector Inteligente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys y configuraciones con manejo de errores
try:
    GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", "")
except:
    GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")

try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    DEEPSEEK_BASE_URL = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
except:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

@dataclass
class PricingPlan:
    name: str
    price_monthly: float
    exams_limit: int
    features: List[str]
    can_create_groups: bool

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
            "OCR Google avanzado",
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
            "OCR Google Premium",
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
            "OCR Google Enterprise",
            "Integración API",
            "Análisis institucional",
            "Soporte dedicado"
        ],
        can_create_groups=True
    )
}

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

class ImprovedGoogleOCR:
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_available = False
        self.error_message = None
        if not api_key or len(api_key) < 30:
            self.error_message = "Google Vision API key no configurada o inválida"
            st.warning("⚠️ Google Vision API key no configurada o inválida.")
            return
        self.vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        self.is_available = True

    def is_configured(self):
        return self.is_available and self.api_key and len(self.api_key) > 30

    def extract_text_from_image_debug(self, image_data):
        if not self.is_configured():
            return None, "Google Vision API no configurada"
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            request_payload = {
                "requests": [
                    {
                        "image": {"content": image_base64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}],
                        "imageContext": {"languageHints": ["es", "en"]}
                    }
                ]
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.vision_url, headers=headers, json=request_payload, timeout=60)
            if response.status_code != 200:
                return None, f"Error Google Vision API: {response.status_code} - {response.text}"
            result = response.json()
            vision_response = result['responses'][0]
            if 'error' in vision_response:
                error_msg = vision_response['error'].get('message', 'Error desconocido')
                return None, f"Error en Google Vision: {error_msg}"
            if 'textAnnotations' not in vision_response or not vision_response['textAnnotations']:
                return None, "No se detectó texto en la imagen."
            full_text = vision_response['textAnnotations'][0].get('description', '')
            if not full_text.strip():
                return None, "El texto extraído está vacío"
            confidence_info = {
                'avg_confidence': 0.8,
                'quality_ratio': 0.8,
                'total_lines': len(full_text.split('\n')),
                'low_confidence_lines': 0,
                'message': f"Texto extraído exitosamente: {len(full_text)} caracteres"
            }
            return full_text, confidence_info
        except Exception as e:
            return None, f"Error en OCR: {str(e)}"

class DatabaseManager:
    def __init__(self):
        self.init_database()
    def init_database(self):
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
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
                text_quality REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (group_id) REFERENCES groups (id)
            )
        ''')
        conn.commit()
        conn.close()
    def reset_monthly_limits(self):
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
        self.client = None
        self.db = None
        self.google_ocr = None
        self.initialization_errors = []
        try:
            self.db = DatabaseManager()
        except Exception as e:
            self.initialization_errors.append(f"Error base de datos: {str(e)}")
            st.error(f"Error al inicializar base de datos: {str(e)}")
        try:
            if not DEEPSEEK_API_KEY:
                self.initialization_errors.append("DeepSeek API key no configurada")
                st.error("❌ DeepSeek API key no configurada. Por favor configura DEEPSEEK_API_KEY en los secrets o variables de entorno.")
            else:
                self.client = openai.OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL
                )
        except Exception as e:
            self.initialization_errors.append(f"Error DeepSeek API: {str(e)}")
            st.error(f"Error al inicializar DeepSeek API: {str(e)}")
        try:
            if GOOGLE_VISION_API_KEY:
                self.google_ocr = ImprovedGoogleOCR(GOOGLE_VISION_API_KEY)
                if not self.google_ocr.is_configured():
                    self.initialization_errors.append("Google Vision API no configurada correctamente")
            else:
                self.initialization_errors.append("Google Vision API key no encontrada")
                st.warning("⚠️ Google Vision API no configurada. Funcionalidad OCR limitada.")
        except Exception as e:
            self.initialization_errors.append(f"Error Google OCR: {str(e)}")
            st.error(f"Error al inicializar Google OCR: {str(e)}")

    def extract_text_from_file(self, uploaded_file):
        try:
            file_type = uploaded_file.type
            ocr_method = "unknown"
            text_quality = 0.0
            text = None
            if file_type == "application/pdf":
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    if page_text.strip() and len(page_text.strip()) > 50:
                        text += page_text
                        ocr_method = "pdf_native"
                        text_quality = 0.9
                    else:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("png")
                        ocr_text, _ = self.google_ocr.extract_text_from_image_debug(img_data)
                        if ocr_text and len(ocr_text.strip()) > 10:
                            text += ocr_text + "\n"
                            ocr_method = "google_ocr"
                            text_quality = 0.7
                        else:
                            text += "[Página sin texto reconocido]\n"
                            text_quality = 0.3
                pdf_document.close()
            elif file_type.startswith("image/"):
                image_bytes = uploaded_file.read()
                if self.google_ocr and self.google_ocr.is_configured():
                    extracted_text, confidence_info = self.google_ocr.extract_text_from_image_debug(image_bytes)
                    if extracted_text:
                        text = extracted_text
                        ocr_method = "google_ocr"
                        text_quality = confidence_info.get('avg_confidence', 0.5)
                    else:
                        text = None
                        ocr_method = "failed"
                        text_quality = 0.0
                        st.error(f"Error OCR: {confidence_info}")
                else:
                    text = None
                    ocr_method = "no_ocr"
                    text_quality = 0.0
            else:
                text = None
                ocr_method = "unsupported_format"
                text_quality = 0.0
                st.error(f"Formato no soportado: {file_type}")
            return text, ocr_method, text_quality
        except Exception as e:
            st.error(f"Error extrayendo texto: {str(e)}")
            return None, "error", 0.0

    def correct_exam(self, text, subject, rubric=None, total_points=10):
        if not self.client:
            return None, "DeepSeek API no configurada"
        try:
            system_prompt = f"""
            Eres un profesor experto en {subject} con años de experiencia en corrección de exámenes.
            TAREA: Corregir el siguiente examen de manera objetiva y constructiva.
            CRITERIOS DE EVALUACIÓN:
            - Puntuación total: {total_points} puntos
            - Materia: {subject}
            {"- Rúbrica específica: " + rubric if rubric else ""}
            FORMATO DE RESPUESTA (JSON):
            {{
                "puntuacion_total": float,
                "puntuacion_maxima": {total_points},
                "porcentaje": float,
                "calificacion_letra": "A/B/C/D/F",
                "preguntas_analizadas": [],
                "resumen_general": {{
                    "fortalezas": [],
                    "areas_mejora": [],
                    "recomendaciones": []
                }},
                "tiempo_estimado_estudio": "",
                "recursos_recomendados": []
            }}
            INSTRUCCIONES:
            1. Analiza cada pregunta individualmente
            2. Asigna puntuación parcial cuando sea apropiado
            3. Explica claramente por qué cada respuesta es correcta o incorrecta
            4. Proporciona sugerencias constructivas
            5. Mantén un tono profesional y alentador
            6. Si no puedes identificar preguntas claramente, analiza el contenido general
            """
            user_prompt = f"EXAMEN A CORREGIR:\n{text}\nPor favor, corrige este examen siguiendo los criterios especificados y devuelve la evaluación en formato JSON."
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            response_text = response.choices[0].message.content
            try:
                correction_data = json.loads(response_text)
                return correction_data, None
            except json.JSONDecodeError:
                return {
                    "puntuacion_total": 0,
                    "puntuacion_maxima": total_points,
                    "porcentaje": 0,
                    "calificacion_letra": "F",
                    "preguntas_analizadas": [],
                    "resumen_general": {
                        "fortalezas": [],
                        "areas_mejora": ["Error en el procesamiento de la respuesta"],
                        "recomendaciones": ["Revisar el texto del examen"]
                    },
                    "respuesta_raw": response_text
                }, "Error parseando JSON"
        except Exception as e:
            return None, f"Error en corrección: {str(e)}"

    def save_exam_result(self, user_id, group_id, filename, subject, correction_data, ocr_method, text_quality):
        if not self.db:
            return False
        try:
            conn = sqlite3.connect('mentor_ia.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO exams (user_id, group_id, filename, subject, grade, total_points, corrections, ocr_method, text_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                group_id,
                filename,
                subject,
                correction_data.get('puntuacion_total', 0),
                correction_data.get('puntuacion_maxima', 10),
                json.dumps(correction_data),
                ocr_method,
                text_quality
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error guardando resultado: {str(e)}")
            return False

    def get_user_stats(self, user_id):
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
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO groups (user_id, name, subject, description)
            VALUES (?, ?, ?, ?)
        ''', (user_id, name, subject, description))
        conn.commit()
        conn.close()

    def get_user_groups(self, user_id):
        conn = sqlite3.connect('mentor_ia.db')
        df_groups = pd.read_sql_query('''
            SELECT * FROM groups WHERE user_id = ? ORDER BY created_at DESC
        ''', conn, params=(user_id,))
        conn.close()
        return df_groups

    def update_user_plan(self, user_id, plan):
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET plan = ? WHERE id = ?
        ''', (plan, user_id))
        conn.commit()
        conn.close()

def show_ocr_configuration():
    st.subheader("🔧 Configuración Google Vision OCR")
    corrector = st.session_state.get('corrector')
    if corrector and hasattr(corrector, 'google_ocr') and corrector.google_ocr.is_configured():
        st.success("✅ Google Vision OCR configurado correctamente")
        st.info("API Key: " + "*" * 20 + corrector.google_ocr.api_key[-4:])
        st.subheader("🧪 Probar OCR")
        test_image = st.file_uploader("Sube una imagen para probar OCR", type=['png', 'jpg', 'jpeg'])
        if test_image and st.button("Probar OCR"):
            with st.spinner("Procesando imagen..."):
                image_data = test_image.read()
                text, info = corrector.google_ocr.extract_text_from_image_debug(image_data)
                if text:
                    st.success("✅ OCR funcionando correctamente")
                    st.text_area("Texto extraído:", text, height=200)
                else:
                    st.error(f"❌ Error en OCR: {info}")
    else:
        st.error("❌ Google Vision OCR no configurado")
        st.markdown("""
        **Para configurar Google Vision OCR:**
        1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
        2. Habilita la API de Vision
        3. Obtén tu API Key
        4. Actualiza la variable `GOOGLE_VISION_API_KEY` en tus secrets o variables de entorno
        """)
           
def show_pricing():
    """Muestra página de precios"""
    st.title("💰 Planes y Precios")
    
    # Mostrar planes en columnas
    cols = st.columns(len(PRICING_PLANS))
    
    for i, (plan_key, plan) in enumerate(PRICING_PLANS.items()):
        with cols[i]:
            # Destacar plan recomendado
            if plan_key == "basic":
                st.markdown("### 🌟 " + plan.name)
                st.markdown("*Recomendado*")
            else:
                st.markdown("### " + plan.name)
            
            # Precio
            if plan.price_monthly == 0:
                st.markdown("## **GRATIS**")
            else:
                st.markdown(f"## **${plan.price_monthly:.2f}**/mes")
            
            # Características
            st.markdown("**Características:**")
            for feature in plan.features:
                st.markdown(f"✅ {feature}")
            
            # Botón de selección
            if st.button(f"Seleccionar {plan.name}", key=f"select_{plan_key}"):
                user = st.session_state.get('user')
                if user:
                    corrector = st.session_state['corrector']
                    corrector.update_user_plan(user[0], plan_key)
                    st.success(f"Plan actualizado a {plan.name}")
                    st.rerun()

def show_groups_management():
    """Muestra gestión de grupos"""
    st.title("👥 Gestión de Grupos")
    
    user = st.session_state.get('user')
    if not user:
        st.error("Usuario no encontrado")
        return
    
    corrector = st.session_state['corrector']
    user_plan = PRICING_PLANS.get(user[2], PRICING_PLANS['free'])
    
    if not user_plan.can_create_groups:
        st.warning("⚠️ Necesitas un plan premium para crear grupos")
        st.info("Los grupos te permiten organizar exámenes por clase o asignatura")
        return
    
    # Crear nuevo grupo
    st.subheader("➕ Crear Nuevo Grupo")
    
    with st.form("new_group_form"):
        group_name = st.text_input("Nombre del grupo", placeholder="Ej: Matemáticas 3°A")
        group_subject = st.selectbox("Asignatura", list(SUBJECT_COLORS.keys()))
        group_description = st.text_area("Descripción (opcional)", placeholder="Descripción del grupo...")
        
        if st.form_submit_button("Crear Grupo"):
            if group_name.strip():
                corrector.create_group(user[0], group_name.strip(), group_subject, group_description.strip())
                st.success(f"Grupo '{group_name}' creado exitosamente")
                st.rerun()
            else:
                st.error("El nombre del grupo es obligatorio")
    
    # Mostrar grupos existentes
    st.subheader("📋 Mis Grupos")
    
    df_groups = corrector.get_user_groups(user[0])
    
    if df_groups.empty:
        st.info("No tienes grupos creados aún")
    else:
        for _, group in df_groups.iterrows():
            with st.expander(f"📁 {group['name']} ({group['subject']})"):
                st.write(f"**Asignatura:** {group['subject']}")
                st.write(f"**Creado:** {group['created_at']}")
                if group['description']:
                    st.write(f"**Descripción:** {group['description']}")
                
                # Estadísticas del grupo
                conn = sqlite3.connect('mentor_ia.db')
                group_stats = pd.read_sql_query('''
                    SELECT COUNT(*) as total_exams, AVG(grade) as avg_grade
                    FROM exams WHERE group_id = ?
                ''', conn, params=(group['id'],))
                conn.close()
                
                if group_stats.iloc[0]['total_exams'] > 0:
                    st.write(f"**Exámenes:** {group_stats.iloc[0]['total_exams']}")
                    st.write(f"**Promedio:** {group_stats.iloc[0]['avg_grade']:.1f}")

def show_statistics():
    """Muestra estadísticas detalladas"""
    st.title("📊 Estadísticas y Análisis")
    
    user = st.session_state.get('user')
    if not user:
        st.error("Usuario no encontrado")
        return
    
    corrector = st.session_state['corrector']
    df_exams = corrector.get_user_stats(user[0])
    
    if df_exams.empty:
        st.info("No hay exámenes para mostrar estadísticas")
        return
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Exámenes", len(df_exams))
    
    with col2:
        avg_grade = df_exams['grade'].mean()
        st.metric("Promedio General", f"{avg_grade:.1f}")
    
    with col3:
        last_exam = df_exams.iloc[0] if not df_exams.empty else None
        if last_exam is not None:
            st.metric("Último Examen", f"{last_exam['grade']:.1f}")
    
    with col4:
        user_plan = PRICING_PLANS.get(user[2], PRICING_PLANS['free'])
        remaining = user_plan.exams_limit - user[3]
        st.metric("Exámenes Restantes", remaining)
    
    # Gráfico de evolución
    st.subheader("📈 Evolución de Calificaciones")
    
    if len(df_exams) > 1:
        df_exams_sorted = df_exams.sort_values('created_at')
        
        fig = px.line(
            df_exams_sorted, 
            x='created_at', 
            y='grade',
            color='subject',
            title="Evolución de Calificaciones por Tiempo",
            labels={'grade': 'Calificación', 'created_at': 'Fecha'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribución por asignatura
    st.subheader("📚 Distribución por Asignatura")
    
    subject_stats = df_exams.groupby('subject').agg({
        'grade': ['count', 'mean', 'std']
    }).round(2)
    
    subject_stats.columns = ['Cantidad', 'Promedio', 'Desv. Estándar']
    st.dataframe(subject_stats)
    
    # Gráfico de barras por asignatura
    fig_bar = px.bar(
        df_exams.groupby('subject')['grade'].mean().reset_index(),
        x='subject',
        y='grade',
        title="Promedio por Asignatura",
        color='subject',
        color_discrete_map=SUBJECT_COLORS
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Calidad de OCR
    st.subheader("🔍 Calidad de OCR")
    
    ocr_quality = df_exams.groupby('ocr_method').agg({
        'text_quality': 'mean',
        'grade': 'count'
    }).round(3)
    
    ocr_quality.columns = ['Calidad Promedio', 'Cantidad']
    st.dataframe(ocr_quality)
    
    # Exámenes recientes
    st.subheader("📋 Exámenes Recientes")
    
    recent_exams = df_exams.head(10)[['filename', 'subject', 'grade', 'created_at', 'ocr_method']]
    st.dataframe(recent_exams)

def show_help():
    """Muestra página de ayuda"""
    st.title("❓ Ayuda y Soporte")
    
    st.markdown("""
    ## 🚀 Cómo usar Mentor.ia
    
    ### 1. Subir Archivos
    - **Imágenes**: PNG, JPG, JPEG (máximo 20MB)
    - **PDFs**: Con texto o imágenes
    - **Texto**: Archivos .txt
    
    ### 2. Configurar Evaluación
    - Selecciona la asignatura
    - Define criterios de evaluación
    - Establece la rúbrica
    
    ### 3. Procesamiento
    - El sistema usa OCR para extraer texto
    - IA DeepSeek evalúa el contenido
    - Genera calificación y comentarios
    
    ## 🔧 Configuración OCR
    
    ### Google Vision OCR (Recomendado)
    - Precisión avanzada en escritura manual
    - Reconocimiento de caracteres mejorado
    - Procesamiento rápido y seguro
    
    ### Configuración:
    1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
    2. Habilita la API de Vision
    3. Obtén tu API Key
    4. Actualiza la variable `GOOGLE_VISION_API_KEY` en tus secrets o variables de entorno
    
    ## 💡 Consejos para Mejores Resultados
    
    ### Calidad de Imagen
    - Usa buena iluminación
    - Evita sombras y reflejos
    - Letra clara y legible
    - Resolución alta (mínimo 1080p)
    
    ### Escritura
    - Letra clara y espaciada
    - Tinta oscura sobre papel blanco
    - Evita correcciones excesivas
    - Organiza las respuestas
    
    ## 📊 Planes y Límites
    
    ### Plan Gratuito
    - 25 exámenes/mes
    - Funcionalidades básicas
    - Sin grupos
    
    ### Planes Premium
    - Más exámenes mensuales
    - Creación de grupos
    - OCR avanzado
    - Estadísticas detalladas
    
    ## 🐛 Solución de Problemas
    
    ### "No se pudo extraer texto"
    - Verificar calidad de imagen
    - Mejorar iluminación
    - Usar mayor resolución
    - Verificar configuración OCR
    
    ### "Texto extraído muy corto"
    - Imagen puede estar borrosa
    - Letra demasiado pequeña
    - Contraste insuficiente
    
    ### "Error en corrección"
    - Problema de conectividad
    - Límites de API excedidos
    - Contenido no válido
    
    ## 📧 Contacto
    
    Para soporte técnico o consultas:
    - Email: soporte@mentor-ia.com
    - Documentación: docs.mentor-ia.com
    - Estado del servicio: status.mentor-ia.com
    """)

def main():
    """Función principal de la aplicación"""
    
    # Inicializar corrector
    if 'corrector' not in st.session_state:
        st.session_state['corrector'] = ExamCorrector()
    
    corrector = st.session_state['corrector']
    
    # Resetear límites mensuales
    corrector.db.reset_monthly_limits()
    
    # Obtener o crear usuario
    if 'user' not in st.session_state:
        st.session_state['user'] = corrector.get_or_create_user()
    
    user = st.session_state['user']
    user_plan = PRICING_PLANS.get(user[2], PRICING_PLANS['free'])
    
    # Sidebar con navegación
    with st.sidebar:
        st.title("🎓 Mentor.ia")
        st.markdown("---")
        
        # Información del usuario
        st.subheader("👤 Usuario")
        st.write(f"**Plan:** {user_plan.name}")
        st.write(f"**Exámenes usados:** {user[3]}/{user_plan.exams_limit}")
        
        # Barra de progreso
        usage_pct = (user[3] / user_plan.exams_limit) * 100
        st.progress(usage_pct / 100)
        
        st.markdown("---")
        
        # Navegación
        page = st.selectbox(
            "Navegar",
            ["🏠 Inicio", "📝 Corrector", "👥 Grupos", "📊 Estadísticas", "💰 Planes", "🔧 Configuración", "❓ Ayuda"]
        )
    
    # Contenido principal según página seleccionada
    if page == "🏠 Inicio":
        show_home()
    elif page == "📝 Corrector":
        show_corrector()
    elif page == "👥 Grupos":
        show_groups_management()
    elif page == "📊 Estadísticas":
        show_statistics()
    elif page == "💰 Planes":
        show_pricing()
    elif page == "🔧 Configuración":
        show_ocr_configuration()
    elif page == "❓ Ayuda":
        show_help()

def show_home():
    """Página de inicio"""
    st.title("🎓 Mentor.ia - Corrector Inteligente")
    
    st.markdown("""
    ## Bienvenido a Mentor.ia
    
    Tu asistente inteligente para la corrección automática de exámenes.
    
    ### ✨ Características principales:
    - **OCR Avanzado**: Extrae texto de imágenes y PDFs
    - **IA DeepSeek**: Evaluación inteligente del contenido
    - **Múltiples Formatos**: Soporta imágenes, PDFs y texto
    - **Estadísticas**: Análisis detallado del rendimiento
    - **Grupos**: Organiza exámenes por clase o asignatura
    
    ### 🚀 Empezar:
    1. Ve a **📝 Corrector** para evaluar un examen
    2. Configura **👥 Grupos** para organizar tus clases
    3. Revisa **📊 Estadísticas** para análisis detallado
    """)
    
    # Estadísticas rápidas
    user = st.session_state.get('user')
    if user:
        corrector = st.session_state['corrector']
        df_exams = corrector.get_user_stats(user[0])
        
        if not df_exams.empty:
            st.subheader("📊 Resumen Rápido")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Exámenes Totales", len(df_exams))
            
            with col2:
                avg_grade = df_exams['grade'].mean()
                st.metric("Promedio General", f"{avg_grade:.1f}")
            
            with col3:
                last_exam = df_exams.iloc[0] if not df_exams.empty else None
                if last_exam is not None:
                    st.metric("Último Examen", f"{last_exam['grade']:.1f}")

def show_corrector():
    """Página principal del corrector"""
    st.title("📝 Corrector Inteligente")
    
    user = st.session_state.get('user')
    if not user:
        st.error("Usuario no encontrado")
        return
    
    corrector = st.session_state['corrector']
    user_plan = PRICING_PLANS.get(user[2], PRICING_PLANS['free'])
    
    # Verificar límites
    if user[3] >= user_plan.exams_limit:
        st.error(f"Has alcanzado el límite de {user_plan.exams_limit} exámenes para tu plan {user_plan.name}")
        st.info("Actualiza tu plan para continuar corrigiendo exámenes")
        return
    
    # Configuración del examen
    st.subheader("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        subject = st.selectbox(
            "Asignatura",
            list(SUBJECT_COLORS.keys()),
            help="Selecciona la asignatura del examen"
        )
        group_id = None
        if user_plan.can_create_groups:
            df_groups = corrector.get_user_groups(user[0])
            if not df_groups.empty:
                group_options = ["Sin grupo"] + df_groups['name'].tolist()
                group_selection = st.selectbox("Grupo", group_options)
                if group_selection != "Sin grupo":
                    group_id = df_groups[df_groups['name'] == group_selection]['id'].iloc[0]
    with col2:
        evaluation_mode = st.radio(
            "Modo de Evaluación",
            ["Automático", "Personalizado"],
            help="Automático: criterios generados por IA | Personalizado: define tus propios criterios"
        )
    if evaluation_mode == "Personalizado":
        st.subheader("📋 Criterios de Evaluación")
        col1, col2 = st.columns(2)
        with col1:
            criteria = st.text_area(
                "Criterios de Evaluación",
                placeholder="Ej: Comprensión conceptual, aplicación de fórmulas, claridad en la explicación...",
                height=100
            )
        with col2:
            rubric = st.text_area(
                "Rúbrica de Calificación",
                placeholder="Ej: Excelente (90-100), Bueno (70-89), Regular (50-69), Deficiente (0-49)",
                height=100
            )
    st.subheader("📤 Subir Examen")
    uploaded_file = st.file_uploader(
        "Selecciona el archivo del examen",
        type=['png', 'jpg', 'jpeg', 'pdf', 'txt'],
        help="Formatos soportados: PNG, JPG, JPEG, PDF, TXT"
    )
    if uploaded_file is not None:
        st.success(f"Archivo cargado: {uploaded_file.name}")
        
        # Lee el archivo SOLO UNA VEZ
        file_bytes = uploaded_file.read()
        
        # Mostrar preview si es imagen
        if uploaded_file.type.startswith('image/'):
            st.image(Image.open(BytesIO(file_bytes)), caption="Vista previa", use_column_width=True)
        
        # Botón para procesar
        if st.button("🚀 Procesar Examen", type="primary"):
            with st.spinner("Procesando examen..."):
                # Extraer texto usando los bytes leídos
                text, ocr_method, text_quality = None, None, None
                if uploaded_file.type.startswith('image/'):
                    if corrector.google_ocr and corrector.google_ocr.is_configured():
                        text, info = corrector.google_ocr.extract_text_from_image_debug(file_bytes)
                        ocr_method = "google_ocr"
                        text_quality = info.get('avg_confidence', 0.5) if info else 0.0
                    else:
                        st.error("OCR no configurado")
                        return
                elif uploaded_file.type == "application/pdf":
                    # Si es PDF, puedes adaptar tu lógica aquí usando file_bytes
                    text, ocr_method, text_quality = corrector.extract_text_from_file(BytesIO(file_bytes))
                else:
                    text, ocr_method, text_quality = corrector.extract_text_from_file(BytesIO(file_bytes))
                
                if text is None:
                    st.error("No se pudo extraer texto del archivo")
                    return
                
                # Mostrar texto extraído
                with st.expander("📄 Texto Extraído"):
                    st.text_area("Contenido:", text, height=200)
                    st.info(f"Método: {ocr_method} | Calidad: {text_quality:.2f}")
                
                # Generar criterios automáticamente si es necesario
                if evaluation_mode == "Automático":
                    st.info("Generando criterios automáticamente...")
                    auto_criteria = corrector.generate_criteria_from_text(text, subject)
                    criteria = auto_criteria['criteria']
                    rubric = auto_criteria['rubric']
                    
                    with st.expander("📋 Criterios Generados"):
                        st.write("**Criterios:**", criteria)
                        st.write("**Rúbrica:**", rubric)
                
                # Corregir examen
                st.info("Evaluando con IA...")
                result = corrector.correct_exam(text, criteria, rubric, subject)
                
                if result:
                    # Mostrar resultados
                    show_exam_results(result)
                    
                    # Guardar resultado
                    corrector.save_exam_result(
                        user[0], 
                        group_id, 
                        uploaded_file.name, 
                        subject, 
                        result, 
                        ocr_method, 
                        text_quality
                    )
                    
                    # Actualizar usuario en sesión
                    st.session_state['user'] = corrector.get_or_create_user(user[1])
                    
                    st.success("✅ Examen procesado y guardado exitosamente")
                else:
                    st.error("Error al procesar el examen")

def show_exam_results(result):
    """Muestra los resultados de la corrección"""
    st.subheader("📊 Resultados de la Corrección")
    
    # Nota principal
    nota_final = result['nota_final']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Calificación Final",
            f"{nota_final['puntuacion']:.1f}",
            f"{nota_final['porcentaje']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Letra",
            nota_final['letra']
        )
    
    with col3:
        st.metric(
            "Puntaje",
            f"{nota_final['puntuacion']}/{nota_final['puntuacion_maxima']}"
        )
    
    # Evaluaciones detalladas
    if 'evaluaciones' in result:
        st.subheader("📋 Evaluación Detallada")
        
        for evaluacion in result['evaluaciones']:
            with st.expander(f"📝 {evaluacion['seccion']}"):
                
                # Puntuación de la sección
                st.write(f"**Puntuación:** {evaluacion['puntos']}/{evaluacion['max_puntos']}")
                
                # Comentario
                if evaluacion.get('comentario'):
                    st.write(f"**Comentario:** {evaluacion['comentario']}")
                
                # Fortalezas
                if evaluacion.get('fortalezas'):
                    st.write("**Fortalezas:**")
                    for fortaleza in evaluacion['fortalezas']:
                        st.write(f"✅ {fortaleza}")
                
                # Mejoras
                if evaluacion.get('mejoras'):
                    st.write("**Áreas de Mejora:**")
                    for mejora in evaluacion['mejoras']:
                        st.write(f"📈 {mejora}")
    
    # Comentario general
    if result.get('comentario'):
        st.subheader("💬 Comentario General")
        st.write(result['comentario'])
    
    # Recomendaciones
    if result.get('recomendaciones'):
        st.subheader("💡 Recomendaciones")
        for recomendacion in result['recomendaciones']:
            st.write(f"• {recomendacion}")
    
    # Información de calidad
    if result.get('calidad_texto'):
        st.subheader("🔍 Calidad del Procesamiento")
        st.info(f"Calidad del texto extraído: {result['calidad_texto']}")

if __name__ == "__main__":
    main()

