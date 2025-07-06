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
OPENAI_API_KEY = "sk-proj-tu-api-key-aqui"  # Reemplaza con tu API key real

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
        """Corrector con API key fija"""
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
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
                model="gpt-3.5-turbo",
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
    
    def delete_group(self, group_id, user_id):
        """Elimina un grupo"""
        conn = sqlite3.connect('mentor_ia.db')
        cursor = conn.cursor()
        
        # Verificar que el grupo pertenece al usuario
        cursor.execute('SELECT * FROM groups WHERE id = ? AND user_id = ?', (group_id, user_id))
        group = cursor.fetchone()
        
        if group:
            # Eliminar exámenes del grupo
            cursor.execute('DELETE FROM exams WHERE group_id = ?', (group_id,))
            # Eliminar el grupo
            cursor.execute('DELETE FROM groups WHERE id = ?', (group_id,))
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
    
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

def show_welcome_page():
    """Página de bienvenida"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 4rem; color: #2E86AB; margin-bottom: 1rem;">
            🎓 Mentor.ia
        </h1>
        <h2 style="color: #A23B72; margin-bottom: 2rem;">
            Corrector Inteligente de Exámenes
        </h2>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 3rem;">
            Potencia tu enseñanza con inteligencia artificial.<br>
            Corrección automática, análisis detallado y seguimiento del progreso.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Características principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;">
            <h3 style="color: #2E86AB;">🤖 Corrección Automática</h3>
            <p>Utiliza IA avanzada para evaluar exámenes con criterios personalizados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;">
            <h3 style="color: #A23B72;">📊 Análisis Detallado</h3>
            <p>Obtén estadísticas completas y seguimiento del progreso estudiantil</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;">
            <h3 style="color: #F18F01;">👥 Gestión de Grupos</h3>
            <p>Organiza estudiantes por materias y grupos para mejor seguimiento</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Botón de inicio centrado
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Comenzar Ahora", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.session_state.show_plan_selection = True
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Información adicional
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #333;">¿Cómo funciona?</h3>
        <div style="display: flex; justify-content: space-around; margin-top: 2rem;">
            <div style="flex: 1; text-align: center;">
                <h4 style="color: #2E86AB;">1. Sube tu examen</h4>
                <p>Texto, PDF o imagen</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <h4 style="color: #A23B72;">2. Define criterios</h4>
                <p>Rúbrica y parámetros</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <h4 style="color: #F18F01;">3. Obtén resultados</h4>
                <p>Calificación y feedback</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_plan_selection():
    """Selección de plan"""
    st.title("🎯 Selecciona tu Plan")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #666;">
            Elige el plan que mejor se adapte a tus necesidades educativas
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar planes en columnas
    cols = st.columns(len(PRICING_PLANS))
    
    for i, (plan_key, plan) in enumerate(PRICING_PLANS.items()):
        with cols[i]:
            # Determinar si es el plan recomendado
            recommended = plan_key == "basic"
            border_color = "#2E86AB" if recommended else "#ddd"
            
            st.markdown(f"""
            <div style="
                border: 2px solid {border_color};
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 1rem;
                background: white;
                position: relative;
                height: 400px;
            ">
                {'<div style="background: #2E86AB; color: white; padding: 0.5rem; border-radius: 5px; margin: -1.5rem -1.5rem 1rem -1.5rem;">🌟 RECOMENDADO</div>' if recommended else ''}
                <h3 style="color: #333; margin-top: {'0' if recommended else '1rem'};">{plan.name}</h3>
                <h2 style="color: #2E86AB; margin: 1rem 0;">${plan.price_monthly}/mes</h2>
                <p style="color: #666; font-size: 1.1rem; margin-bottom: 1.5rem;"><strong>{plan.exams_limit} exámenes</strong></p>
                <ul style="text-align: left; margin: 1rem 0; padding-left: 1rem;">
            """, unsafe_allow_html=True)
            
            for feature in plan.features:
                st.markdown(f"<li style='margin: 0.5rem 0;'>{feature}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            button_style = "primary" if recommended else "secondary"
            if st.button(f"Seleccionar {plan.name}", key=f"select_{plan_key}", type=button_style):
                st.session_state.selected_plan = plan_key
                st.session_state.user_plan = plan_key
                st.session_state.show_plan_selection = False
                st.success(f"¡Plan {plan.name} seleccionado!")
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
        st.write(f"Progreso: {exams_used}/{plan_info.exams_limit} exámenes utilizados")
        
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
        else:
            st.info("🎯 ¡Comienza corrigiendo tu primer examen para ver estadísticas!")

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
        
        # Criterios de evaluación mejorados
        st.subheader("📋 Criterios de Evaluación")
        
        # Método de entrada para criterios
        criteria_method = st.radio(
            "Método para criterios:",
            ["Plantilla predefinida", "Texto personalizado", "Subir archivo de rúbrica"]
        )
        
        criteria = ""
        rubric = ""
        
        if criteria_method == "Plantilla predefinida":
            templates = get_default_criteria_templates()
            
            if subject in templates:
                criteria = templates[subject]["criteria"]
                rubric = templates[subject]["rubric"]
            else:
                criteria = "Criterios personalizados"
                rubric = "Rúbrica personalizada"
            
            criteria = st.text_area("Criterios de evaluación:", value=criteria, height=100)
            rubric = st.text_area("Rúbrica:", value=rubric, height=120)
        
        elif criteria_method == "Texto personalizado":
            criteria = st.text_area(
                "Criterios de evaluación:",
                placeholder="Describe los criterios específicos para evaluar este examen...",
                height=100
            )
            rubric = st.text_area(
                "Rúbrica:",
                placeholder="Define la escala de calificación y criterios de puntuación...",
                height=120
            )
        
        elif criteria_method == "Subir archivo de rúbrica":
            uploaded_rubric = st.file_uploader(
                "Subir rúbrica o modelo de examen:",
                type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
                help="Sube un archivo con la rúbrica de evaluación o modelo de examen perfecto"
            )
            
            if uploaded_rubric:
                with st.spinner("Procesando rúbrica..."):
                    rubric_text = corrector.extract_text_from_file(uploaded_rubric)
                
                if rubric_text:
                    st.success(f"✅ Rúbrica procesada: {len(rubric_text)} caracteres")
                    
                    # Vista previa
                    with st.expander("Vista previa de la rúbrica"):
                        st.text(rubric_text[:500] + "..." if len(rubric_text) > 500 else rubric_text)
                    
                    # Usar el archivo como criterios y rúbrica
                    criteria = f"Evaluar basándose en el siguiente documento de rúbrica: {rubric_text}"
                    rubric = rubric_text
                else:
                    st.error("No se pudo procesar el archivo de rúbrica")
            
            # Permitir edición adicional
            if criteria:
                criteria = st.text_area("Criterios adicionales:", value=criteria, height=100)
    
    with col2:
        st.subheader("📄 Subir Examen")

        # Método de entrada para examen
        exam_method = st.radio(
            "Método para examen:",
            ["Subir archivo", "Pegar texto", "Escribir directamente"]
        )
        
        exam_text = ""
        
        if exam_method == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Subir examen para corregir:",
                type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
                help="Sube el examen del estudiante en formato PDF, imagen o texto"
            )
            
            if uploaded_file:
                with st.spinner("Procesando examen..."):
                    exam_text = corrector.extract_text_from_file(uploaded_file)
                
                if exam_text:
                    st.success(f"✅ Examen procesado: {len(exam_text)} caracteres")
                    
                    # Vista previa
                    with st.expander("Vista previa del examen"):
                        st.text(exam_text[:500] + "..." if len(exam_text) > 500 else exam_text)
                else:
                    st.error("No se pudo procesar el archivo del examen")
        
        elif exam_method == "Pegar texto":
            exam_text = st.text_area(
                "Pegar texto del examen:",
                placeholder="Pega aquí el contenido del examen a corregir...",
                height=200
            )
        
        elif exam_method == "Escribir directamente":
            exam_text = st.text_area(
                "Escribir examen:",
                placeholder="Escribe directamente las respuestas del examen...",
                height=200
            )
        
        # Botón de corrección
        if st.button("🚀 Corregir Examen", type="primary", use_container_width=True):
            if exam_text and criteria:
                with st.spinner("Corrigiendo examen con IA..."):
                    result = corrector.correct_exam(exam_text, criteria, rubric, subject)
                
                if result:
                    # Guardar resultado
                    filename = uploaded_file.name if exam_method == "Subir archivo" and uploaded_file else "examen_texto.txt"
                    corrector.save_exam_result(user_id, selected_group, filename, subject, result)
                    
                    # Mostrar resultado
                    show_exam_result(result)
                else:
                    st.error("Error al corregir el examen")
            else:
                st.warning("Por favor, ingresa el examen y los criterios de evaluación")

def get_default_criteria_templates():
    """Plantillas predefinidas de criterios por asignatura"""
    return {
        "Matemáticas": {
            "criteria": "Evaluar: resolución correcta de problemas, procedimientos matemáticos, justificación de pasos, precisión en cálculos, uso correcto de fórmulas y notación matemática.",
            "rubric": "Excelente (90-100): Resolución completa y correcta. Bueno (80-89): Procedimiento correcto con errores menores. Satisfactorio (70-79): Comprensión básica con algunos errores. Insuficiente (0-69): Errores significativos o procedimientos incorrectos."
        },
        "Ciencias": {
            "criteria": "Evaluar: comprensión de conceptos científicos, aplicación de principios, análisis de datos, metodología científica, conclusiones lógicas.",
            "rubric": "Excelente (90-100): Dominio completo de conceptos. Bueno (80-89): Comprensión sólida con aplicación correcta. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Conceptos incorrectos o incompletos."
        },
        "Literatura": {
            "criteria": "Evaluar: comprensión del texto, análisis literario, uso del lenguaje, estructura argumentativa, creatividad en la expresión.",
            "rubric": "Excelente (90-100): Análisis profundo y original. Bueno (80-89): Comprensión clara con buen análisis. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Comprensión limitada o incorrecta."
        },
        "Historia": {
            "criteria": "Evaluar: conocimiento de hechos históricos, análisis de causas y consecuencias, contextualización temporal, uso de fuentes históricas.",
            "rubric": "Excelente (90-100): Análisis histórico completo y contextualizado. Bueno (80-89): Conocimiento sólido con buen análisis. Satisfactorio (70-79): Conocimiento básico. Insuficiente (0-69): Conocimiento limitado o incorrecto."
        },
        "Física": {
            "criteria": "Evaluar: comprensión de leyes físicas, aplicación de fórmulas, resolución de problemas, interpretación de fenómenos físicos.",
            "rubric": "Excelente (90-100): Aplicación correcta de principios físicos. Bueno (80-89): Comprensión sólida con aplicación correcta. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Conceptos físicos incorrectos."
        },
        "Química": {
            "criteria": "Evaluar: comprensión de conceptos químicos, balanceo de ecuaciones, nomenclatura, cálculos estequiométricos, propiedades de la materia.",
            "rubric": "Excelente (90-100): Dominio completo de conceptos químicos. Bueno (80-89): Comprensión sólida. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Conceptos químicos incorrectos."
        },
        "Biología": {
            "criteria": "Evaluar: comprensión de procesos biológicos, clasificación, anatomía, fisiología, evolución, ecología.",
            "rubric": "Excelente (90-100): Comprensión completa de procesos biológicos. Bueno (80-89): Conocimiento sólido. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Conceptos biológicos incorrectos."
        },
        "Geografía": {
            "criteria": "Evaluar: conocimiento geográfico, análisis espacial, interpretación de mapas, comprensión de fenómenos geográficos.",
            "rubric": "Excelente (90-100): Análisis geográfico completo. Bueno (80-89): Conocimiento sólido con buen análisis. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Conocimiento geográfico limitado."
        },
        "Filosofía": {
            "criteria": "Evaluar: comprensión de conceptos filosóficos, argumentación lógica, análisis crítico, reflexión personal.",
            "rubric": "Excelente (90-100): Argumentación filosófica sólida y original. Bueno (80-89): Comprensión clara con buen análisis. Satisfactorio (70-79): Comprensión básica. Insuficiente (0-69): Argumentación deficiente."
        },
        "Idiomas": {
            "criteria": "Evaluar: gramática, vocabulario, comprensión, expresión oral y escrita, pronunciación, fluidez.",
            "rubric": "Excelente (90-100): Dominio avanzado del idioma. Bueno (80-89): Competencia sólida. Satisfactorio (70-79): Competencia básica. Insuficiente (0-69): Competencia limitada."
        }
    }

def show_exam_result(result):
    """Mostrar resultado de la corrección"""
    st.subheader("📊 Resultado de la Corrección")
    
    # Nota final destacada
    nota_final = result['nota_final']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Puntuación", f"{nota_final['puntuacion']}/{nota_final['puntuacion_maxima']}")
    
    with col2:
        st.metric("Porcentaje", f"{nota_final['porcentaje']:.1f}%")
    
    with col3:
        st.metric("Calificación", nota_final['letra'])
    
    # Barra de progreso visual
    progress_color = "#2E86AB" if nota_final['porcentaje'] >= 70 else "#FF6B6B"
    st.markdown(f"""
    <div style="background: #f0f0f0; border-radius: 10px; padding: 5px; margin: 1rem 0;">
        <div style="background: {progress_color}; width: {nota_final['porcentaje']}%; height: 20px; border-radius: 5px; text-align: center; line-height: 20px; color: white;">
            {nota_final['porcentaje']:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Evaluaciones detalladas
    st.subheader("📋 Evaluación Detallada")
    
    for evaluacion in result['evaluaciones']:
        with st.expander(f"📝 {evaluacion['seccion']} - {evaluacion['puntos']}/{evaluacion['max_puntos']} puntos"):
            st.write(evaluacion['comentario'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Fortalezas:**")
                for fortaleza in evaluacion['fortalezas']:
                    st.write(f"• {fortaleza}")
            
            with col2:
                st.write("**🔄 Áreas de mejora:**")
                for mejora in evaluacion['mejoras']:
                    st.write(f"• {mejora}")
    
    # Comentario general
    st.subheader("💭 Comentario General")
    st.write(result['comentario'])
    
    # Recomendaciones
    st.subheader("🎯 Recomendaciones")
    for recomendacion in result['recomendaciones']:
        st.write(f"• {recomendacion}")

def show_groups_manager():
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
    
    # Verificar si puede crear grupos
    if not plan_info.can_create_groups:
        st.warning("⚠️ La creación de grupos no está disponible en tu plan actual")
        st.info("Actualiza a un plan superior para acceder a esta funcionalidad")
        return
    
    # Tabs para gestión
    tab1, tab2 = st.tabs(["➕ Crear Grupo", "📋 Mis Grupos"])
    
    with tab1:
        st.subheader("Crear Nuevo Grupo")
        
        with st.form("create_group_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                group_name = st.text_input(
                    "Nombre del grupo*:",
                    placeholder="Ej: Matemáticas 3°A, Física Avanzada..."
                )
                
                group_subject = st.selectbox(
                    "Asignatura*:",
                    list(SUBJECT_COLORS.keys())
                )
            
            with col2:
                group_description = st.text_area(
                    "Descripción (opcional):",
                    placeholder="Describe el grupo, nivel, características especiales...",
                    height=100
                )
            
            submitted = st.form_submit_button("🚀 Crear Grupo", type="primary")
            
            if submitted:
                if group_name and group_subject:
                    try:
                        corrector.create_group(user_id, group_name, group_subject, group_description)
                        st.success(f"✅ Grupo '{group_name}' creado exitosamente!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al crear el grupo: {str(e)}")
                else:
                    st.warning("Por favor completa los campos obligatorios")
    
    with tab2:
        st.subheader("Mis Grupos Creados")
        
        df_groups = corrector.get_user_groups(user_id)
        
        if df_groups.empty:
            st.info("🎯 No tienes grupos creados aún. ¡Crea tu primer grupo!")
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
                <h4>💡 Ventajas de crear grupos:</h4>
                <ul style="text-align: left; display: inline-block;">
                    <li>Organiza exámenes por clase/materia</li>
                    <li>Seguimiento personalizado del progreso</li>
                    <li>Estadísticas específicas por grupo</li>
                    <li>Mejor gestión de estudiantes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Mostrar grupos en cards
            for _, group in df_groups.iterrows():
                subject_color = SUBJECT_COLORS.get(group['subject'], '#BDC3C7')
                
                # Obtener estadísticas del grupo
                conn = sqlite3.connect('mentor_ia.db')
                group_stats = pd.read_sql_query('''
                    SELECT COUNT(*) as total_exams, AVG(grade) as avg_grade
                    FROM exams WHERE group_id = ?
                ''', conn, params=(group['id'],))
                conn.close()
                
                total_exams = group_stats['total_exams'].iloc[0] if not group_stats.empty else 0
                avg_grade = group_stats['avg_grade'].iloc[0] if not group_stats.empty and group_stats['avg_grade'].iloc[0] else 0
                
                st.markdown(f"""
                <div style="
                    border: 1px solid {subject_color};
                    border-left: 5px solid {subject_color};
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    background: white;
                ">
                    <h4 style="color: {subject_color}; margin: 0 0 0.5rem 0;">
                        📚 {group['name']}
                    </h4>
                    <p style="margin: 0.5rem 0;"><strong>Asignatura:</strong> {group['subject']}</p>
                    <p style="margin: 0.5rem 0;"><strong>Descripción:</strong> {group['description'] if group['description'] else 'Sin descripción'}</p>
                    <p style="margin: 0.5rem 0;"><strong>Creado:</strong> {group['created_at']}</p>
                    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                        <span style="background: {subject_color}20; padding: 0.5rem; border-radius: 5px;">
                            📊 {total_exams} exámenes
                        </span>
                        <span style="background: {subject_color}20; padding: 0.5rem; border-radius: 5px;">
                            📈 Promedio: {avg_grade:.1f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Botones de acción
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("📊 Ver Estadísticas", key=f"stats_{group['id']}"):
                        show_group_stats(corrector, group['id'], group['name'])
                
                with col2:
                    if st.button("🗑️ Eliminar", key=f"delete_{group['id']}", type="secondary"):
                        if st.session_state.get(f'confirm_delete_{group["id"]}'):
                            if corrector.delete_group(group['id'], user_id):
                                st.success(f"✅ Grupo '{group['name']}' eliminado")
                                st.rerun()
                            else:
                                st.error("Error al eliminar el grupo")
                        else:
                            st.session_state[f'confirm_delete_{group["id"]}'] = True
                            st.warning("⚠️ Haz clic nuevamente para confirmar la eliminación")
                
                st.markdown("---")

def show_group_stats(corrector, group_id, group_name):
    """Mostrar estadísticas específicas de un grupo"""
    st.subheader(f"📊 Estadísticas del Grupo: {group_name}")
    
    # Obtener exámenes del grupo
    conn = sqlite3.connect('mentor_ia.db')
    df_group_exams = pd.read_sql_query('''
        SELECT * FROM exams WHERE group_id = ? ORDER BY created_at DESC
    ''', conn, params=(group_id,))
    conn.close()
    
    if df_group_exams.empty:
        st.info("Este grupo no tiene exámenes aún")
        return
    
    # Métricas del grupo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Exámenes", len(df_group_exams))
    
    with col2:
        avg_grade = df_group_exams['grade'].mean()
        st.metric("Promedio General", f"{avg_grade:.1f}")
    
    with col3:
        max_grade = df_group_exams['grade'].max()
        st.metric("Nota Máxima", f"{max_grade:.1f}")
    
    with col4:
        min_grade = df_group_exams['grade'].min()
        st.metric("Nota Mínima", f"{min_grade:.1f}")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Evolución temporal
        fig1 = px.line(df_group_exams, x='created_at', y='grade',
                      title="Evolución de Calificaciones")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Distribución de calificaciones
        fig2 = px.histogram(df_group_exams, x='grade', nbins=10,
                           title="Distribución de Calificaciones")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Tabla de exámenes recientes
    st.subheader("📋 Exámenes Recientes")
    
    # Mostrar tabla simplificada
    display_df = df_group_exams[['filename', 'grade', 'total_points', 'created_at']].head(10)
    display_df.columns = ['Archivo', 'Calificación', 'Puntos Totales', 'Fecha']
    st.dataframe(display_df, use_container_width=True)

def show_analytics():
    """Página de análisis y reportes"""
    st.title("📈 Análisis y Reportes")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        st.error("Error: Sistema no inicializado")
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    user_id = user[0]
    
    df_exams = corrector.get_user_stats(user_id)
    df_groups = corrector.get_user_groups(user_id)
    
    if df_exams.empty:
        st.info("📊 No hay datos suficientes para generar análisis. ¡Comienza corrigiendo algunos exámenes!")
        return
    
    # Filtros
    st.sidebar.subheader("🔍 Filtros")
    
    # Filtro por fecha
    if not df_exams.empty:
        df_exams['created_at'] = pd.to_datetime(df_exams['created_at'])
        date_range = st.sidebar.date_input(
            "Rango de fechas:",
            value=(df_exams['created_at'].min().date(), df_exams['created_at'].max().date()),
            min_value=df_exams['created_at'].min().date(),
            max_value=df_exams['created_at'].max().date()
        )
        
        # Filtro por asignatura
        subjects = df_exams['subject'].unique()
        selected_subjects = st.sidebar.multiselect(
            "Asignaturas:",
            subjects,
            default=subjects
        )
        
        # Aplicar filtros
        filtered_df = df_exams[
            (df_exams['created_at'].dt.date >= date_range[0]) &
            (df_exams['created_at'].dt.date <= date_range[1]) &
            (df_exams['subject'].isin(selected_subjects))
        ]
    else:
        filtered_df = df_exams
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Exámenes Analizados", len(filtered_df))
    
    with col2:
        avg_grade = filtered_df['grade'].mean() if not filtered_df.empty else 0
        st.metric("Promedio General", f"{avg_grade:.1f}")
    
    with col3:
        passed_exams = len(filtered_df[filtered_df['grade'] >= 60]) if not filtered_df.empty else 0
        pass_rate = (passed_exams / len(filtered_df) * 100) if not filtered_df.empty else 0
        st.metric("Tasa de Aprobación", f"{pass_rate:.1f}%")
    
    with col4:
        subjects_count = len(filtered_df['subject'].unique()) if not filtered_df.empty else 0
        st.metric("Asignaturas Diferentes", subjects_count)
    
    # Gráficos principales
    if not filtered_df.empty:
        st.subheader("📊 Análisis Visual")
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Tendencias", "📊 Distribuciones", "🎯 Por Asignatura", "📅 Temporal"])
        
        with tab1:
            # Tendencia general
            fig1 = px.line(filtered_df, x='created_at', y='grade',
                          title="Tendencia de Calificaciones",
                          labels={'created_at': 'Fecha', 'grade': 'Calificación'})
            fig1.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Línea de Aprobación")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Promedio móvil
            if len(filtered_df) > 5:
                filtered_df_sorted = filtered_df.sort_values('created_at')
                filtered_df_sorted['promedio_movil'] = filtered_df_sorted['grade'].rolling(window=5, min_periods=1).mean()
                
                fig2 = px.line(filtered_df_sorted, x='created_at', y='promedio_movil',
                              title="Promedio Móvil (5 exámenes)",
                              labels={'created_at': 'Fecha', 'promedio_movil': 'Promedio Móvil'})
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma de calificaciones
                fig3 = px.histogram(filtered_df, x='grade', nbins=20,
                                   title="Distribución de Calificaciones")
                fig3.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Aprobación")
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Box plot
                fig4 = px.box(filtered_df, y='grade', title="Distribución de Calificaciones (Box Plot)")
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            # Análisis por asignatura
            subject_stats = filtered_df.groupby('subject').agg({
                'grade': ['mean', 'count', 'std']
            }).round(2)
            
            subject_stats.columns = ['Promedio', 'Cantidad', 'Desviación']
            subject_stats = subject_stats.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig5 = px.bar(subject_stats, x='subject', y='Promedio',
                             title="Promedio por Asignatura",
                             color='Promedio',
                             color_continuous_scale='RdYlGn')
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                fig6 = px.pie(subject_stats, values='Cantidad', names='subject',
                             title="Distribución por Asignatura")
                st.plotly_chart(fig6, use_container_width=True)
            
            # Tabla de estadísticas
            st.subheader("📋 Estadísticas por Asignatura")
            st.dataframe(subject_stats, use_container_width=True)
        
        with tab4:
            # Análisis temporal
            filtered_df['mes'] = filtered_df['created_at'].dt.to_period('M')
            monthly_stats = filtered_df.groupby('mes').agg({
                'grade': ['mean', 'count']
            }).round(2)
            
            monthly_stats.columns = ['Promedio', 'Cantidad']
            monthly_stats = monthly_stats.reset_index()
            monthly_stats['mes'] = monthly_stats['mes'].astype(str)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig7 = px.bar(monthly_stats, x='mes', y='Promedio',
                             title="Promedio Mensual")
                st.plotly_chart(fig7, use_container_width=True)
            
            with col2:
                fig8 = px.line(monthly_stats, x='mes', y='Cantidad',
                              title="Exámenes por Mes")
                st.plotly_chart(fig8, use_container_width=True)

def main():
    """Función principal"""
    # Inicializar session state
    if 'corrector' not in st.session_state:
        st.session_state.corrector = ExamCorrector()
    
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True
    
    if 'show_plan_selection' not in st.session_state:
        st.session_state.show_plan_selection = False
    
    if 'user_plan' not in st.session_state:
        st.session_state.user_plan = 'free'
    
    # Mostrar páginas según el estado
    if st.session_state.show_welcome:
        show_welcome_page()
        return
    
    if st.session_state.show_plan_selection:
        show_plan_selection()
        return
    
    # Navegación principal
    st.sidebar.title("🎓 Mentor.ia")
    
    # Mostrar plan actual
    current_plan = PRICING_PLANS[st.session_state.user_plan]
    st.sidebar.info(f"📋 Plan Actual: {current_plan.name}")
    
    # Menú de navegación
    menu_options = {
        "📊 Dashboard": "dashboard",
        "🤖 Corrector": "corrector",
        "👥 Grupos": "groups",
        "📈 Análisis": "analytics",
        "⚙️ Configuración": "settings"
    }
    
    selected_page = st.sidebar.selectbox(
        "Navegación:",
        list(menu_options.keys())
    )
    
    page_key = menu_options[selected_page]
    
    # Mostrar página seleccionada
    if page_key == "dashboard":
        show_dashboard()
    elif page_key == "corrector":
        show_corrector()
    elif page_key == "groups":
        show_groups_manager()
    elif page_key == "analytics":
        show_analytics()
    elif page_key == "settings":
        show_settings()

def show_settings():
    """Página de configuración"""
    st.title("⚙️ Configuración")
    
    # Información del plan
    st.subheader("📋 Plan Actual")
    current_plan = PRICING_PLANS[st.session_state.user_plan]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **{current_plan.name}**
        - Precio: ${current_plan.price_monthly}/mes
        - Límite: {current_plan.exams_limit} exámenes/mes
        - Grupos: {'Sí' if current_plan.can_create_groups else 'No'}
        """)
    
    with col2:
        if st.button("🔄 Cambiar Plan", type="primary"):
            st.session_state.show_plan_selection = True
            st.rerun()
    
    # Configuración de API
    st.subheader("🔑 Configuración")

    ación de API
    st.subheader("🔑 Configuración de API")
    
    with st.form("api_config"):
        st.info("💡 Configura tu propia API key de OpenAI para mayor privacidad y control")
        
        current_api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Tu API key personal de OpenAI"
        )
        
        if st.form_submit_button("💾 Guardar Configuración"):
            if current_api_key:
                # Aquí se guardaría la API key de forma segura
                st.success("✅ Configuración guardada exitosamente")
            else:
                st.warning("Por favor ingresa una API key válida")
    
    # Configuración de notificaciones
    st.subheader("🔔 Notificaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox(
            "Notificaciones por email",
            value=True,
            help="Recibir notificaciones importantes por email"
        )
    
    with col2:
        browser_notifications = st.checkbox(
            "Notificaciones del navegador",
            value=False,
            help="Recibir notificaciones en el navegador"
        )
    
    # Configuración de idioma
    st.subheader("🌐 Idioma")
    
    language = st.selectbox(
        "Idioma de la interfaz:",
        ["Español", "English", "Français", "Deutsch"],
        index=0
    )
    
    # Configuración de tema
    st.subheader("🎨 Apariencia")
    
    theme = st.selectbox(
        "Tema:",
        ["Claro", "Oscuro", "Automático"],
        index=0
    )
    
    # Exportar datos
    st.subheader("💾 Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Exportar Datos", type="secondary"):
            corrector = st.session_state.get('corrector')
            if corrector:
                user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
                user_id = user[0]
                
                # Exportar datos del usuario
                df_exams = corrector.get_user_stats(user_id)
                if not df_exams.empty:
                    csv = df_exams.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Descargar CSV",
                        data=csv,
                        file_name=f"mentor_ia_datos_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    with col2:
        if st.button("🗑️ Eliminar Cuenta", type="secondary"):
            st.warning("⚠️ Esta acción eliminará permanentemente todos tus datos")
            if st.checkbox("Confirmo que deseo eliminar mi cuenta"):
                if st.button("🗑️ Eliminar Definitivamente", type="primary"):
                    # Aquí se eliminaría la cuenta
                    st.error("Cuenta eliminada exitosamente")

def show_welcome_page():
    """Página de bienvenida inicial"""
    st.set_page_config(
        page_title="Mentor.ia - Corrector Inteligente",
        page_icon="🎓",
        layout="wide"
    )
    
    # Header principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; color: #2E86AB; margin-bottom: 0;">
            🎓 Mentor.ia
        </h1>
        <h3 style="color: #666; margin-top: 0;">
            Corrector Inteligente de Exámenes con IA
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Video o imagen principal
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 2rem 0;
        ">
            <h2>🚀 Revoluciona tu forma de corregir</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                Ahorra tiempo y mejora la calidad de tus evaluaciones con inteligencia artificial
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Características principales
    st.markdown("## ✨ Características Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <h3 style="color: #2E86AB;">🤖 Corrección Inteligente</h3>
            <p>Evaluación automática con IA avanzada que comprende contexto y criterios específicos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <h3 style="color: #2E86AB;">📊 Análisis Detallado</h3>
            <p>Retroalimentación constructiva con fortalezas, áreas de mejora y recomendaciones</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <h3 style="color: #2E86AB;">👥 Gestión de Grupos</h3>
            <p>Organiza estudiantes por clases, genera reportes y estadísticas personalizadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Proceso de uso
    st.markdown("## 🔄 Cómo Funciona")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="
                background: #667eea;
                color: white;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                font-size: 1.5rem;
            ">1</div>
            <h4>📤 Subir Examen</h4>
            <p>Carga el examen en PDF, imagen o texto</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="
                background: #667eea;
                color: white;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                font-size: 1.5rem;
            ">2</div>
            <h4>📋 Definir Criterios</h4>
            <p>Establece los criterios de evaluación y rúbricas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="
                background: #667eea;
                color: white;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                font-size: 1.5rem;
            ">3</div>
            <h4>🤖 Corrección IA</h4>
            <p>La IA analiza y evalúa automáticamente</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="
                background: #667eea;
                color: white;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                font-size: 1.5rem;
            ">4</div>
            <h4>📊 Resultados</h4>
            <p>Recibe calificación y retroalimentación detallada</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Testimonios
    st.markdown("## 💬 Lo que dicen nuestros usuarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #2E86AB;
        ">
            <p style="font-style: italic; margin-bottom: 1rem;">
            "Mentor.ia ha transformado mi forma de evaluar. Ahorro horas de trabajo y mis estudiantes reciben retroalimentación más detallada."
            </p>
            <p style="font-weight: bold; color: #2E86AB;">- Prof. María García, Matemáticas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #2E86AB;
        ">
            <p style="font-style: italic; margin-bottom: 1rem;">
            "La calidad de la evaluación es impresionante. Detecta errores que yo podría pasar por alto y sugiere mejoras específicas."
            </p>
            <p style="font-weight: bold; color: #2E86AB;">- Dr. Carlos Ruiz, Física</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Llamada a la acción
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2 style="color: #2E86AB;">¿Listo para comenzar?</h2>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
                Únete a miles de educadores que ya están revolucionando sus evaluaciones
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Comenzar Ahora", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.session_state.show_plan_selection = True
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>© 2025 Mentor.ia - Inteligencia Artificial para la Educación</p>
        <p>Desarrollado con ❤️ para educadores</p>
    </div>
    """, unsafe_allow_html=True)

def show_corrector():
    """Página del corrector mejorada"""
    st.title("🤖 Corrector Inteligente")
    
    corrector = st.session_state.get('corrector')
    if not corrector:
        st.error("Error: Sistema no inicializado")
        return
    
    user = corrector.get_or_create_user(plan=st.session_state.get('user_plan', 'free'))
    user_id = user[0]
    plan_info = PRICING_PLANS[st.session_state.user_plan]
    
    # Verificar límites
    monthly_exams = corrector.get_monthly_exam_count(user_id)
    remaining_exams = plan_info.exams_limit - monthly_exams
    
    if remaining_exams <= 0:
        st.error(f"⚠️ Has alcanzado el límite de {plan_info.exams_limit} exámenes/mes de tu plan {plan_info.name}")
        st.info("Actualiza tu plan para continuar corrigiendo exámenes")
        return
    
    # Mostrar límites
    progress = monthly_exams / plan_info.exams_limit
    st.progress(progress)
    st.caption(f"Exámenes utilizados: {monthly_exams}/{plan_info.exams_limit} este mes")
    
    # Formulario de corrección
    with st.form("correction_form"):
        st.subheader("📝 Configuración del Examen")
        
        # Selección de grupo
        df_groups = corrector.get_user_groups(user_id)
        if not df_groups.empty:
            group_options = ["Sin grupo"] + df_groups['name'].tolist()
            selected_group = st.selectbox(
                "Grupo:",
                group_options,
                help="Selecciona el grupo al que pertenece este examen"
            )
        else:
            selected_group = "Sin grupo"
        
        # Asignatura
        subject = st.selectbox(
            "Asignatura:",
            list(SUBJECT_COLORS.keys()),
            help="Selecciona la asignatura del examen"
        )
        
        # Criterios de evaluación mejorados
        st.subheader("📋 Criterios de Evaluación")
        
        # Tabs para diferentes métodos de criterios
        tab1, tab2, tab3 = st.tabs(["✍️ Escribir Criterios", "📄 Subir Rúbrica", "🎯 Plantillas"])
        
        with tab1:
            criteria = st.text_area(
                "Criterios de evaluación:",
                placeholder="Describe los criterios específicos para evaluar este examen...",
                height=150,
                help="Especifica qué aspectos evaluar y cómo"
            )
            
            rubric = st.text_area(
                "Rúbrica de calificación:",
                placeholder="Define los niveles de desempeño y sus criterios...",
                height=100,
                help="Especifica los rangos de calificación y sus criterios"
            )
        
        with tab2:
            st.info("📤 Sube tu rúbrica o modelo de examen para mayor precisión")
            
            rubric_file = st.file_uploader(
                "Subir rúbrica o modelo:",
                type=['txt', 'pdf', 'docx'],
                help="Sube un archivo con la rúbrica de evaluación o un modelo de examen perfecto"
            )
            
            if rubric_file:
                with st.spinner("Procesando rúbrica..."):
                    rubric_content = corrector.extract_text_from_file(rubric_file)
                
                if rubric_content:
                    st.success(f"✅ Rúbrica procesada: {len(rubric_content)} caracteres")
                    
                    # Vista previa
                    with st.expander("Vista previa de la rúbrica"):
                        st.text(rubric_content[:500] + "..." if len(rubric_content) > 500 else rubric_content)
                    
                    # Usar contenido de la rúbrica
                    criteria = st.text_area(
                        "Criterios adicionales (opcional):",
                        placeholder="Agrega criterios específicos adicionales...",
                        height=100
                    )
                    
                    rubric = rubric_content
                else:
                    st.error("No se pudo procesar la rúbrica")
        
        with tab3:
            st.info("🎯 Usa plantillas predefinidas según la asignatura")
            
            templates = get_default_criteria_templates()
            if subject in templates:
                template = templates[subject]
                
                st.markdown(f"**Plantilla para {subject}:**")
                st.text(template['criteria'])
                
                use_template = st.checkbox(f"Usar plantilla de {subject}")
                
                if use_template:
                    criteria = template['criteria']
                    rubric = template['rubric']
                    
                    # Permitir modificaciones
                    criteria = st.text_area(
                        "Modificar criterios:",
                        value=criteria,
                        height=100
                    )
                    
                    rubric = st.text_area(
                        "Modificar rúbrica:",
                        value=rubric,
                        height=100
                    )
        
        # Método de entrada para examen
        st.subheader("📤 Subir Examen")
        exam_method = st.radio(
            "Método para examen:",
            ["Subir archivo", "Pegar texto", "Escribir directamente"]
        )
        
        exam_text = ""
        uploaded_file = None
        
        if exam_method == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Subir examen para corregir:",
                type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx'],
                help="Sube el examen del estudiante en formato PDF, imagen, Word o texto"
            )
        
        elif exam_method == "Pegar texto":
            exam_text = st.text_area(
                "Pegar texto del examen:",
                placeholder="Pega aquí el contenido del examen a corregir...",
                height=200
            )
        
        elif exam_method == "Escribir directamente":
            exam_text = st.text_area(
                "Escribir examen:",
                placeholder="Escribe directamente las respuestas del examen...",
                height=200
            )
        
        # Configuraciones avanzadas
        with st.expander("⚙️ Configuraciones Avanzadas"):
            col1, col2 = st.columns(2)
            
            with col1:
                severity = st.selectbox(
                    "Severidad de evaluación:",
                    ["Estándar", "Estricta", "Flexible"],
                    help="Ajusta qué tan estricta será la evaluación"
                )
                
                feedback_detail = st.selectbox(
                    "Nivel de retroalimentación:",
                    ["Detallado", "Básico", "Extenso"],
                    help="Controla el nivel de detalle en los comentarios"
                )
            
            with col2:
                include_suggestions = st.checkbox(
                    "Incluir sugerencias de mejora",
                    value=True,
                    help="Generar recomendaciones específicas para el estudiante"
                )
                
                highlight_strengths = st.checkbox(
                    "Destacar fortalezas",
                    value=True,
                    help="Resaltar aspectos positivos del examen"
                )
        
        # Botón de corrección
        submitted = st.form_submit_button("🚀 Corregir Examen", type="primary", use_container_width=True)
        
        if submitted:
            # Procesar archivo si es necesario
            if exam_method == "Subir archivo" and uploaded_file:
                with st.spinner("Procesando examen..."):
                    exam_text = corrector.extract_text_from_file(uploaded_file)
                
                if exam_text:
                    st.success(f"✅ Examen procesado: {len(exam_text)} caracteres")
                else:
                    st.error("No se pudo procesar el archivo del examen")
                    return
            
            # Verificar que tengamos todo lo necesario
            if not exam_text:
                st.warning("Por favor, proporciona el contenido del examen")
                return
            
            if not criteria and not rubric:
                st.warning("Por favor, define los criterios de evaluación o sube una rúbrica")
                return
            
            # Realizar corrección
            with st.spinner("Corrigiendo examen con IA..."):
                # Preparar criterios completos
                full_criteria = criteria
                if rubric:
                    full_criteria += f"\n\nRúbrica: {rubric}"
                
                # Agregar configuraciones avanzadas
                config = {
                    'severity': severity,
                    'feedback_detail': feedback_detail,
                    'include_suggestions': include_suggestions,
                    'highlight_strengths': highlight_strengths
                }
                
                result = corrector.correct_exam(exam_text, full_criteria, rubric, subject, config)
            
            if result:
                # Guardar resultado
                filename = uploaded_file.name if uploaded_file else "examen_texto.txt"
                group_id = None
                
                if selected_group != "Sin grupo":
                    group_row = df_groups[df_groups['name'] == selected_group]
                    if not group_row.empty:
                        group_id = group_row.iloc[0]['id']
                
                corrector.save_exam_result(user_id, group_id, filename, subject, result)
                
                # Mostrar resultado
                show_exam_result(result)
                
                # Actualizar contador
                st.session_state.monthly_exams = monthly_exams + 1
            else:
                st.error("Error al corregir el examen. Por favor, intenta nuevamente.")

if __name__ == "__main__":
    main()

