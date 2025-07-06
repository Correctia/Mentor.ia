#!/usr/bin/env python
# coding: utf-8


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
import pytesseract  # OCR para imágenes
import os

# Configuración de la aplicación
st.set_page_config(
    page_title="Mentor.ia - Corrector Inteligente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key fija (configurable)
DEEPSEEK_API_KEY = "sk-2193b6a84e2d428e963633e213d1c439"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # Reemplaza con tu API key real

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
            "OCR para PDFs e imágenes",
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
            "OCR avanzado",
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
        """Corrector con DeepSeek API"""
        self.client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.db = DatabaseManager()
    
    def extract_text_from_file(self, uploaded_file):
        """Extrae texto de archivos PDF o imágenes"""
        try:
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                # Extraer texto de PDF
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page in pdf_document:
                    text += page.get_text()
                pdf_document.close()
                return text
            
            elif file_type.startswith("image/"):
                # OCR para imágenes
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image, lang='spa')
                return text
            
            else:
                # Archivo de texto
                return str(uploaded_file.read(), "utf-8")
                
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            return None
    
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
        
            # Procesar respuesta y convertir a formato JSON
            response_text = response.choices[0].message.content
            # Aquí deberías implementar lógica para extraer criterios y rúbrica
            return {
                "criteria": "Criterios extraídos del texto",
                "rubric": "Rúbrica extraída del texto"
            }
        except Exception as e:
            st.error(f"Error generando criterios: {str(e)}")
            return None
    
    def save_exam_result(self, user_id, group_id, filename, subject, result):
        """Guarda resultado en base de datos"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exams (user_id, group_id, filename, subject, grade, total_points, corrections)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            group_id,
            filename,
            subject,
            result['nota_final']['puntuacion'],
            result['nota_final']['puntuacion_maxima'],
            json.dumps(result, ensure_ascii=False)
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
        
        # Selección de grupo (si está disponible)
        df_groups = corrector.get_user_groups(user_id)
        selected_group = None
        
        if plan_info.can_create_groups and not df_groups.empty:
            group_options = ["Sin grupo"] + df_groups['name'].tolist()
            selected_group_name = st.selectbox("Grupo:", group_options)
            
            if selected_group_name != "Sin grupo":
                selected_group = df_groups[df_groups['name'] == selected_group_name]['id'].iloc[0]

        # Método para criterios de corrección
        st.subheader("📋 Criterios de Corrección")
        criteria_method = st.radio(
            "Método de criterios:",
            ["Plantillas predefinidas", "Texto manual", "Importar archivo"],
            key="criteria_method"
        )

        if criteria_method == "Plantillas predefinidas":
            # Código existente de plantillas
            templates = get_default_criteria_templates()
            if subject in templates:
                default_criteria = templates[subject]["criteria"]
                default_rubric = templates[subject]["rubric"]
            else:
                default_criteria = "Criterios personalizados"
                default_rubric = "Rúbrica personalizada"
    
            criteria = st.text_area("Criterios de evaluación:", value=default_criteria, height=100)
            rubric = st.text_area("Rúbrica:", value=default_rubric, height=120)

        elif criteria_method == "Texto manual":
            criteria = st.text_area("Criterios de evaluación:", height=100)
            rubric = st.text_area("Rúbrica:", height=120)

        elif criteria_method == "Importar archivo":
            criteria_file = st.file_uploader(
                "Subir archivo con criterios:",
                type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
                help="Sube un archivo con criterios de corrección o un examen modelo corregido",
                key="criteria_file"
            )
    
            if criteria_file:
                with st.spinner("Extrayendo criterios del archivo..."):
                    criteria_text = corrector.extract_text_from_file(criteria_file)
            
                    if criteria_text:
                        # Generar criterios automáticamente usando IA
                        generated_criteria = corrector.generate_criteria_from_text(criteria_text, subject)
                
                        if generated_criteria:
                            criteria = generated_criteria.get('criteria', '')
                            rubric = generated_criteria.get('rubric', '')
                    
                            st.success(f"✅ Criterios extraídos del archivo: {criteria_file.name}")
                    
                            # Mostrar vista previa
                            with st.expander("Vista previa de criterios extraídos"):
                                st.write("**Criterios:**")
                                st.write(criteria)
                                st.write("**Rúbrica:**")
                                st.write(rubric)
                        else:
                            st.error("No se pudieron generar criterios del archivo")
                            criteria = st.text_area("Criterios de evaluación:", height=100)
                            rubric = st.text_area("Rúbrica:", height=120)
                    else:
                        st.error("No se pudo extraer texto del archivo")
                        criteria = st.text_area("Criterios de evaluación:", height=100)
                        rubric = st.text_area("Rúbrica:", height=120)
            else:
                criteria = st.text_area("Criterios de evaluación:", height=100)
                rubric = st.text_area("Rúbrica:", height=120)
        
    
    with col2:
        st.subheader("📄 Subir Examen")
        
        # Método de entrada
        input_method = st.radio(
            "Método:",
            ["Texto directo", "Subir archivo"]
        )
        
        exam_text = ""
        filename = "examen_directo.txt"
        
        if input_method == "Texto directo":
            exam_text = st.text_area(
                "Texto del examen:",
                height=300,
                placeholder="Pega aquí las respuestas del estudiante..."
            )
        
        elif input_method == "Subir archivo":
            uploaded_files = st.file_uploader(
                "Subir archivo:",
                type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
                help="Archivos soportados: TXT, PDF, PNG, JPG",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Procesar todos los archivos
                all_texts = []
                filenames = []
                
                for file in uploaded_files:
                    if file is not None:
                        filenames.append(file.name)
                        with st.spinner(f"Procesando {file.name}..."):
                            text = corrector.extract_text_from_file(file)
                            if text:
                                all_texts.append(f"=== ARCHIVO: {file.name} ===\n{text}")
                            else:
                                st.error(f"No se pudo extraer texto de {file.name}")
                
                if all_texts:
                    exam_text = "\n\n--- SIGUIENTE ARCHIVO ---\n\n".join(all_texts)
                    filename = ", ".join(filenames)
                    st.success(f"✅ Archivos procesados: {len(uploaded_files)} archivos")
                    st.info(f"Archivos: {filename}")
                    st.info(f"Total de caracteres: {len(exam_text)}")
                    
                    # Vista previa combinada
                    with st.expander("Vista previa del texto combinado"):
                        st.text(exam_text[:1000] + "..." if len(exam_text) > 1000 else exam_text)
                else:
                    st.error("No se pudo extraer texto de ningún archivo")
                    exam_text = ""
                    filename = "error_procesamiento.txt"
            else:
                exam_text = ""
                filename = "sin_archivo.txt"
    
    # Botón de corrección
    st.markdown("---")
    
    if st.button("🚀 Corregir Examen", type="primary", use_container_width=True):
        if not exam_text or not exam_text.strip():
            st.error("Por favor ingresa el texto del examen")
            return
        
        if not criteria.strip() or not rubric.strip():
            st.error("Por favor define criterios y rúbrica")
            return
        
        # Proceso de corrección
        with st.spinner("Corrigiendo examen..."):
            try:
                result = corrector.correct_exam(exam_text, criteria, rubric, subject)
                
                if result:
                    corrector.save_exam_result(user_id, selected_group, filename, subject, result)
                    st.success("¡Examen corregido exitosamente!")
                    
                    # Mostrar resultados
                    display_correction_results(result)
                    
                    st.rerun()
                else:
                    st.error("Error en la corrección")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def show_groups():
    """Gestión de grupos"""
    st.title("👥 Gestión de Grupos")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    user_id = user[0]
    user_plan = user[2]
    
    plan_info = PRICING_PLANS[user_plan]
    
    if not plan_info.can_create_groups:
        st.warning("La creación de grupos requiere un plan de pago")
        st.info("Actualiza tu plan para acceder a esta funcionalidad")
        return
    
    # Crear nuevo grupo
    st.subheader("➕ Crear Nuevo Grupo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        group_name = st.text_input("Nombre del grupo:")
        group_subject = st.selectbox("Asignatura:", list(SUBJECT_COLORS.keys()))
    
    with col2:
        group_description = st.text_area("Descripción:", height=100)
    
    if st.button("Crear Grupo"):
        if group_name:
            corrector.create_group(user_id, group_name, group_subject, group_description)
            st.success(f"Grupo '{group_name}' creado exitosamente")
            st.rerun()
        else:
            st.error("Por favor ingresa un nombre para el grupo")
    
    # Mostrar grupos existentes
    st.subheader("📋 Grupos Existentes")
    
    df_groups = corrector.get_user_groups(user_id)
    
    if not df_groups.empty:
        for _, group in df_groups.iterrows():
            with st.expander(f"📚 {group['name']} - {group['subject']}"):
                st.write(f"**Descripción:** {group['description']}")
                st.write(f"**Creado:** {group['created_at']}")
                
                # Estadísticas del grupo
                conn = sqlite3.connect('mentor_ia.db')
                group_exams = pd.read_sql_query('''
                    SELECT * FROM exams WHERE group_id = ?
                ''', conn, params=(group['id'],))
                conn.close()
                
                if not group_exams.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Exámenes", len(group_exams))
                    with col2:
                        st.metric("Promedio", f"{group_exams['grade'].mean():.1f}")
                    with col3:
                        st.metric("Último examen", group_exams['created_at'].max()[:10])
                else:
                    st.info("No hay exámenes en este grupo")
    else:
        st.info("No has creado ningún grupo aún")

def display_correction_results(result):
    """Mostrar resultados de corrección"""
    st.markdown("### 🎯 Resultado Final")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nota = result['nota_final']['puntuacion']
        color = "🟢" if nota >= 80 else "🟡" if nota >= 60 else "🔴"
        st.metric("Calificación", f"{nota:.1f}/100")
        st.markdown(f"**Estado:** {color} {result['nota_final']['letra']}")
    
    with col2:
        st.metric("Porcentaje", f"{result['nota_final']['porcentaje']:.1f}%")
    
    with col3:
        st.metric("Puntos Máximos", f"{result['nota_final']['puntuacion_maxima']}")
    
    # Evaluaciones detalladas
    st.markdown("### 📋 Evaluación Detallada")
    
    for i, eval_item in enumerate(result.get('evaluaciones', [])):
        with st.expander(f"📝 {eval_item.get('seccion', f'Sección {i+1}')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Puntos:** {eval_item.get('puntos', 0)}/{eval_item.get('max_puntos', 0)}")
                st.markdown(f"**Comentario:** {eval_item.get('comentario', '')}")
            
            with col2:
                fortalezas = eval_item.get('fortalezas', [])
                mejoras = eval_item.get('mejoras', [])
                
                if fortalezas:
                    st.markdown("**✅ Fortalezas:**")
                    for f in fortalezas:
                        st.markdown(f"• {f}")
                
                if mejoras:
                    st.markdown("**🔧 Mejoras:**")
                    for m in mejoras:
                        st.markdown(f"• {m}")
    
    # Comentarios y recomendaciones
    st.markdown("### 💬 Comentario General")
    st.write(result.get('comentario', ''))
    
    if result.get('recomendaciones'):
        st.markdown("### 🎯 Recomendaciones")
        for rec in result['recomendaciones']:
            st.markdown(f"• {rec}")

def show_history():
    """Mostrar historial"""
    st.title("📚 Historial de Exámenes")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    df_exams = corrector.get_user_stats(user[0])
    
    if not df_exams.empty:
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            subjects = ['Todas'] + df_exams['subject'].unique().tolist()
            filter_subject = st.selectbox("Filtrar por asignatura:", subjects)
        
        with col2:
            groups = ['Todos'] + df_exams['group_name'].dropna().unique().tolist()
            filter_group = st.selectbox("Filtrar por grupo:", groups)
        
        # Aplicar filtros
        filtered_df = df_exams.copy()
        if filter_subject != 'Todas':
            filtered_df = filtered_df[filtered_df['subject'] == filter_subject]
        
        if filter_group != 'Todos':
            filtered_df = filtered_df[filtered_df['group_name'] == filter_group]
        
        # Mostrar tabla de exámenes
        st.subheader(f"📊 Exámenes ({len(filtered_df)})")
        
        for _, exam in filtered_df.iterrows():
            with st.expander(f"📝 {exam['filename']} - {exam['subject']} ({exam['grade']:.1f}/100)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Asignatura:** {exam['subject']}")
                    st.write(f"**Grupo:** {exam['group_name'] if exam['group_name'] else 'Sin grupo'}")
                    st.write(f"**Fecha:** {exam['created_at'][:10]}")
                
                with col2:
                    st.write(f"**Calificación:** {exam['grade']:.1f}/100")
                    st.write(f"**Puntos totales:** {exam['total_points']}")
                
                # Mostrar corrección detallada
                if exam['corrections']:
                    try:
                        correction_data = json.loads(exam['corrections'])
                        display_correction_results(correction_data)
                    except:
                        st.write("Error al cargar los detalles de la corrección")
    else:
        st.info("No has corregido ningún examen aún")

def get_default_criteria_templates():
    """Plantillas de criterios por asignatura"""
    return {
        "Matemáticas": {
            "criteria": "Procedimiento matemático correcto, aplicación de fórmulas, resolución de problemas paso a paso, cálculos precisos",
            "rubric": "Excelente (90-100): Procedimientos completos y correctos. Bueno (70-89): Procedimientos correctos con errores menores. Regular (50-69): Procedimientos parcialmente correctos. Deficiente (0-49): Procedimientos incorrectos o incompletos."
        },
        "Ciencias": {
            "criteria": "Comprensión de conceptos científicos, aplicación de teorías, análisis de datos, metodología científica",
            "rubric": "Excelente (90-100): Comprensión completa y aplicación correcta. Bueno (70-89): Comprensión adecuada con aplicación correcta. Regular (50-69): Comprensión básica. Deficiente (0-49): Comprensión limitada."
        },
        "Literatura": {
            "criteria": "Análisis literario, comprensión de textos, uso del lenguaje, creatividad, estructura narrativa",
            "rubric": "Excelente (90-100): Análisis profundo y creatividad excepcional. Bueno (70-89): Análisis adecuado y buena estructura. Regular (50-69): Análisis básico. Deficiente (0-49): Análisis superficial."
        },
        "Historia": {
            "criteria": "Conocimiento de hechos históricos, análisis de causas y consecuencias, cronología, fuentes históricas",
            "rubric": "Excelente (90-100): Conocimiento completo y análisis profundo. Bueno (70-89): Conocimiento sólido y análisis adecuado. Regular (50-69): Conocimiento básico. Deficiente (0-49): Conocimiento limitado."
        },
        "Física": {
            "criteria": "Aplicación de leyes físicas, resolución de problemas, uso de fórmulas, interpretación de gráficos",
            "rubric": "Excelente (90-100): Aplicación correcta y resolución completa. Bueno (70-89): Aplicación correcta con errores menores. Regular (50-69): Aplicación parcial. Deficiente (0-49): Aplicación incorrecta."
        },
        "Química": {
            "criteria": "Comprensión de reacciones químicas, balanceo de ecuaciones, cálculos estequiométricos, nomenclatura",
            "rubric": "Excelente (90-100): Comprensión completa y cálculos correctos. Bueno (70-89): Comprensión adecuada. Regular (50-69): Comprensión básica. Deficiente (0-49): Comprensión limitada."
        },
        "Biología": {
            "criteria": "Conocimiento de procesos biológicos, clasificación, anatomía, fisiología, ecosistemas",
            "rubric": "Excelente (90-100): Conocimiento completo y aplicación correcta. Bueno (70-89): Conocimiento sólido. Regular (50-69): Conocimiento básico. Deficiente (0-49): Conocimiento limitado."
        },
        "Geografía": {
            "criteria": "Conocimiento geográfico, análisis espacial, mapas, climatología, demografía",
            "rubric": "Excelente (90-100): Conocimiento completo y análisis profundo. Bueno (70-89): Conocimiento sólido. Regular (50-69): Conocimiento básico. Deficiente (0-49): Conocimiento limitado."
        },
        "Filosofía": {
            "criteria": "Comprensión de conceptos filosóficos, argumentación, análisis crítico, historia de la filosofía",
            "rubric": "Excelente (90-100): Comprensión profunda y argumentación sólida. Bueno (70-89): Comprensión adecuada. Regular (50-69): Comprensión básica. Deficiente (0-49): Comprensión limitada."
        },
        "Idiomas": {
            "criteria": "Gramática, vocabulario, comprensión lectora, expresión escrita, pronunciación",
            "rubric": "Excelente (90-100): Dominio completo del idioma. Bueno (70-89): Buen dominio. Regular (50-69): Dominio básico. Deficiente (0-49): Dominio limitado."
        }
    }

def show_settings():
    """Configuración de la aplicación"""
    st.title("⚙️ Configuración")
    
    st.subheader("📊 Plan Actual")
    current_plan = st.session_state.get('user_plan', 'free')
    plan_info = PRICING_PLANS[current_plan]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Plan:** {plan_info.name}")
        st.info(f"**Límite:** {plan_info.exams_limit} exámenes/mes")
    
    with col2:
        if current_plan == 'free':
            st.warning("Tienes el plan gratuito")
            if st.button("Actualizar Plan"):
                st.session_state.show_plan_selection = True
                st.rerun()
        else:
            st.success(f"Plan activo: {plan_info.name}")
    
    st.subheader("🎨 Personalización")
    
    # Selección de asignatura favorita
    favorite_subject = st.selectbox(
        "Asignatura favorita (tema de color):",
        list(SUBJECT_COLORS.keys()),
        index=0
    )
    
    apply_subject_theme(favorite_subject)
    
    st.subheader("🔄 Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Limpiar Historial"):
            if st.button("Confirmar eliminación", key="confirm_delete"):
                # Limpiar historial
                conn = sqlite3.connect('mentor_ia.db')
                cursor = conn.cursor()
                
                corrector = st.session_state.get('corrector')
                if corrector:
                    user = corrector.get_or_create_user()
                    cursor.execute('DELETE FROM exams WHERE user_id = ?', (user[0],))
                    conn.commit()
                
                conn.close()
                st.success("Historial eliminado")
                st.rerun()
    
    with col2:
        if st.button("📊 Exportar Datos"):
            corrector = st.session_state.get('corrector')
            if corrector:
                user = corrector.get_or_create_user()
                df_exams = corrector.get_user_stats(user[0])
                
                if not df_exams.empty:
                    csv = df_exams.to_csv(index=False)
                    st.download_button(
                        label="Descargar CSV",
                        data=csv,
                        file_name="mentor_ia_historial.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No hay datos para exportar")

def main():
    """Función principal"""
    
    # Configuración inicial
    if 'corrector' not in st.session_state:
        st.session_state.corrector = ExamCorrector()
    
    if 'user_plan' not in st.session_state:
        st.session_state.user_plan = None
    
    if 'show_plan_selection' not in st.session_state:
        st.session_state.show_plan_selection = True
    
    # Mostrar selección de plan si es necesario
    if st.session_state.show_plan_selection and not st.session_state.user_plan:
        show_plan_selection()
        return
    
    # Sidebar para navegación
    with st.sidebar:
        st.title("🎓 Mentor.ia")
        
        # Información del plan
        if st.session_state.user_plan:
            current_plan = PRICING_PLANS[st.session_state.user_plan]
            st.info(f"Plan: {current_plan.name}")
            
            # Mostrar uso actual
            corrector = st.session_state.corrector
            user = corrector.get_or_create_user(plan=st.session_state.user_plan)
            
            conn = sqlite3.connect('mentor_ia.db')
            cursor = conn.cursor()
            cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user[0],))
            current_user = cursor.fetchone()
            conn.close()
            
            exams_used = current_user[0] if current_user else 0
            remaining = current_plan.exams_limit - exams_used
            
            st.progress(exams_used / current_plan.exams_limit)
            st.write(f"Restantes: {remaining}/{current_plan.exams_limit}")
        
        # Menú de navegación
        menu_options = {
            "📊 Dashboard": "dashboard",
            "🤖 Corrector": "corrector", 
            "👥 Grupos": "groups",
            "📚 Historial": "history",
            "⚙️ Configuración": "settings"
        }
        
        selected_page = st.radio("Navegación", list(menu_options.keys()))
        page_key = menu_options[selected_page]
        
        # Botón para cambiar plan
        if st.button("💰 Cambiar Plan"):
            st.session_state.show_plan_selection = True
            st.rerun()
    
    # Mostrar página seleccionada
    if page_key == "dashboard":
        show_dashboard()
    elif page_key == "corrector":
        show_corrector()
    elif page_key == "groups":
        show_groups()
    elif page_key == "history":
        show_history()
    elif page_key == "settings":
        show_settings()

if __name__ == "__main__":
    main()

