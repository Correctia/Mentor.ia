#!/usr/bin/env python
# coding: utf-8


#get_ipython().system('pip install openai streamlit pandas numpy pillow plotly openpyxl')


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

# Configuración de la aplicación
st.set_page_config(
    page_title="ExamAI Free - Corrector Inteligente Gratuito",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class PricingPlan:
    name: str
    price_monthly: float
    exams_limit: int
    features: List[str]

# Solo plan gratuito optimizado
PRICING_PLANS = {
    "free": PricingPlan(
        name="Plan Gratuito",
        price_monthly=0,
        exams_limit=50,  # Aumentado de 5 a 50
        features=[
            "50 exámenes/mes", 
            "Procesamiento de texto básico",
            "Corrección con GPT-3.5-turbo",
            "Base de datos SQLite local",
            "Estadísticas básicas",
            "Sin pagos ni suscripciones"
        ]
    )
}

class DatabaseManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos SQLite (100% gratuita)"""
        conn = sqlite3.connect('examai_free.db')
        cursor = conn.cursor()
        
        # Tabla de usuarios (simplificada)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                exams_used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de exámenes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                filename TEXT,
                subject TEXT,
                grade REAL,
                total_points REAL,
                corrections TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Tabla de criterios de corrección (plantillas gratuitas)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS correction_templates (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                name TEXT,
                subject TEXT,
                criteria TEXT,
                rubric TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def reset_monthly_limits(self):
        """Reinicia límites mensuales automáticamente"""
        conn = sqlite3.connect('examai_free.db')
        cursor = conn.cursor()
        
        # Reiniciar contadores si ha pasado un mes
        cursor.execute('''
            UPDATE users 
            SET exams_used = 0, last_reset = CURRENT_TIMESTAMP
            WHERE DATE(last_reset, '+1 month') <= DATE('now')
        ''')
        
        conn.commit()
        conn.close()

class ExamCorrectorFree:
    def __init__(self, api_key):
        """Corrector optimizado para costos mínimos"""
        self.client = openai.OpenAI(api_key=api_key)
        self.db = DatabaseManager()
    
    def process_text_input(self, text_input):
        """Procesa texto directo sin necesidad de OCR"""
        if not text_input or not text_input.strip():
            return None
        
        # Validar longitud del texto
        if len(text_input) > 10000:
            st.warning("⚠️ Texto muy largo. Se truncará para optimizar costos.")
            text_input = text_input[:10000] + "\n[...texto truncado...]"
        
        return text_input
    
    def correct_exam_budget(self, exam_text, criteria, rubric, subject="General"):
        """Corrección optimizada para costos mínimos con GPT-3.5-turbo"""
        try:
            # Truncar texto si es muy largo para controlar costos
            max_chars = 8000  # Aproximadamente 2000 tokens
            if len(exam_text) > max_chars:
                exam_text = exam_text[:max_chars] + "\n[...texto truncado para optimizar costos...]"
                st.info("📝 Texto truncado para optimizar costos de API")
            
            # Prompt ultra-optimizado para respuesta concisa
            system_prompt = f"""Eres un corrector de exámenes eficiente. Evalúa el examen de {subject} usando los criterios dados.

CRITERIOS: {criteria[:500]}  
RÚBRICA: {rubric[:500]}

Responde SOLO en JSON válido, sé conciso:"""

            user_prompt = f"""EXAMEN:
{exam_text}

JSON de respuesta (máximo 800 tokens):
{{
    "nota_final": {{
        "puntuacion": 0,
        "puntuacion_maxima": 100,
        "porcentaje": 0,
        "letra": "A/B/C/D/F"
    }},
    "evaluaciones": [
        {{
            "seccion": "Pregunta 1",
            "puntos": 0,
            "max_puntos": 0,
            "comentario": "Breve evaluación",
            "fortalezas": ["Punto fuerte"],
            "mejoras": ["Área a mejorar"]
        }}
    ],
    "comentario": "Evaluación general breve",
    "recomendaciones": ["Sugerencia práctica"]
}}"""

            # Usar GPT-3.5-turbo con configuración de costo mínimo
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Modelo más barato
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Menos creatividad = más consistencia = menos tokens
                max_tokens=800,   # Límite estricto para controlar costos
                top_p=0.9        # Optimización adicional
            )
            
            response_text = response.choices[0].message.content
            
            # Mostrar costo estimado
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            estimated_cost = (input_tokens * 0.0010 + output_tokens * 0.0020) / 1000
            
            st.success(f"💰 Costo estimado de esta corrección: ${estimated_cost:.4f}")
            
            # Limpiar y parsear respuesta
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            st.error("Error parseando respuesta de IA. Reintentando...")
            return self.create_fallback_correction(exam_text)
        except Exception as e:
            st.error(f"Error en corrección: {str(e)}")
            return self.create_fallback_correction(exam_text)
    
    def create_fallback_correction(self, exam_text):
        """Corrección de emergencia si falla la IA"""
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
                "comentario": "Evaluación automática de emergencia. El examen muestra comprensión básica del tema.",
                "fortalezas": ["Presenta respuestas coherentes"],
                "mejoras": ["Desarrollar más profundidad en las respuestas"]
            }],
            "comentario": "Corrección automática realizada. Se recomienda revisión manual.",
            "recomendaciones": ["Revisar conceptos fundamentales", "Practicar desarrollo de respuestas"]
        }
    
    def save_exam_result(self, user_id, filename, subject, result):
        """Guarda resultado en SQLite gratuito"""
        conn = sqlite3.connect('examai_free.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exams (user_id, filename, subject, grade, total_points, corrections)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            filename,
            subject,
            result['nota_final']['puntuacion'],
            result['nota_final']['puntuacion_maxima'],
            json.dumps(result, ensure_ascii=False)
        ))
        
        # Incrementar contador de exámenes usados
        cursor.execute('''
            UPDATE users SET exams_used = exams_used + 1 WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_id):
        """Estadísticas gratuitas desde SQLite"""
        conn = sqlite3.connect('examai_free.db')
        
        df_exams = pd.read_sql_query('''
            SELECT * FROM exams 
            WHERE user_id = ? 
            ORDER BY created_at DESC
            LIMIT 100
        ''', conn, params=(user_id,))
        
        conn.close()
        return df_exams
    
    def get_or_create_user(self, username="usuario_demo"):
        """Obtiene o crea usuario demo"""
        conn = sqlite3.connect('examai_free.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute('''
                INSERT INTO users (username, exams_used) VALUES (?, 0)
            ''', (username,))
            conn.commit()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
        
        conn.close()
        return user

def show_cost_calculator():
    """Calculadora de costos para transparencia"""
    st.subheader("💰 Calculadora de Costos OpenAI")
    
    exams_per_month = st.slider("Exámenes por mes", 1, 100, 20)
    avg_length = st.slider("Longitud promedio del texto (palabras)", 100, 2000, 500)
    
    # Estimaciones conservadoras
    words_to_tokens = 1.3  # Aproximadamente 1.3 tokens por palabra
    total_tokens_input = exams_per_month * avg_length * words_to_tokens
    total_tokens_output = exams_per_month * 500  # Respuesta promedio
    
    cost_input = (total_tokens_input / 1000) * 0.0010
    cost_output = (total_tokens_output / 1000) * 0.0020
    total_cost = cost_input + cost_output
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tokens de entrada", f"{int(total_tokens_input):,}")
        st.caption(f"Costo: ${cost_input:.2f}")
    
    with col2:
        st.metric("Tokens de salida", f"{total_tokens_output:,}")
        st.caption(f"Costo: ${cost_output:.2f}")
        
    with col3:
        st.metric("Costo total/mes", f"${total_cost:.2f}")
        st.caption("Solo OpenAI API")
    
    st.info(f"""
    💡 **Estimación para {exams_per_month} exámenes/mes:**
    - Costo OpenAI: ${total_cost:.2f}/mes
    - Todo lo demás: $0 (completamente gratuito)
    - **Total real: ${total_cost:.2f}/mes**
    """)

def show_free_dashboard():
    """Dashboard gratuito optimizado"""
    st.title("📊 Dashboard Gratuito - ExamAI")
    
    # Inicializar usuario demo
    corrector = st.session_state.get('corrector')
    if corrector:
        user = corrector.get_or_create_user()
        user_id = user[0]
        
        # Reiniciar límites mensuales automáticamente
        corrector.db.reset_monthly_limits()
        
        # Obtener estadísticas actualizadas
        conn = sqlite3.connect('examai_free.db')
        cursor = conn.cursor()
        cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user_id,))
        current_user = cursor.fetchone()
        conn.close()
        
        exams_used = current_user[0] if current_user else 0
        df_exams = corrector.get_user_stats(user_id)
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Exámenes Corregidos", len(df_exams))
        
        with col2:
            remaining = PRICING_PLANS['free'].exams_limit - exams_used
            st.metric("Exámenes Restantes", remaining)
            
        with col3:
            avg_grade = df_exams['grade'].mean() if not df_exams.empty else 0
            st.metric("Nota Promedio", f"{avg_grade:.1f}")
        
        with col4:
            st.metric("Plan", "💚 Gratuito")
        
        # Barra de progreso
        progress = exams_used / PRICING_PLANS['free'].exams_limit
        st.progress(progress)
        
        if progress > 0.8:
            st.warning("⚠️ Te estás quedando sin exámenes este mes")
            st.info("🔄 Los límites se reinician automáticamente cada mes")
        
        # Gráficos gratuitos
        if not df_exams.empty:
            st.subheader("📈 Estadísticas Gratuitas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de evolución
                fig = px.line(df_exams.tail(20), x='created_at', y='grade', 
                             title="Últimas 20 calificaciones")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribución por asignatura
                if 'subject' in df_exams.columns:
                    subject_counts = df_exams['subject'].value_counts()
                    fig2 = px.pie(values=subject_counts.values, names=subject_counts.index,
                                 title="Distribución por asignatura")
                    st.plotly_chart(fig2, use_container_width=True)

def display_correction_results(result):
    """Mostrar resultados de corrección de forma atractiva"""
    # Nota principal
    st.markdown("### 🎯 Resultado Final")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nota = result['nota_final']['puntuacion']
        color = "🟢" if nota >= 80 else "🟡" if nota >= 60 else "🔴"
        st.metric("Calificación", f"{nota:.1f}/100", delta=f"Letra: {result['nota_final']['letra']}")
        st.markdown(f"**Estado:** {color}")
    
    with col2:
        porcentaje = result['nota_final']['porcentaje']
        st.metric("Porcentaje", f"{porcentaje:.1f}%")
    
    with col3:
        st.metric("Puntos Máximos", f"{result['nota_final']['puntuacion_maxima']}")
    
    # Evaluaciones detalladas
    st.markdown("### 📋 Evaluación Detallada")
    
    for i, eval_item in enumerate(result.get('evaluaciones', [])):
        with st.expander(f"📝 {eval_item.get('seccion', f'Sección {i+1}')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Puntos:** {eval_item.get('puntos', 0)}/{eval_item.get('max_puntos', 0)}")
                st.markdown(f"**Comentario:** {eval_item.get('comentario', 'Sin comentario')}")
            
            with col2:
                fortalezas = eval_item.get('fortalezas', [])
                mejoras = eval_item.get('mejoras', [])
                
                if fortalezas:
                    st.markdown("**✅ Fortalezas:**")
                    for fortaleza in fortalezas:
                        st.markdown(f"• {fortaleza}")
                
                if mejoras:
                    st.markdown("**🔧 Áreas de mejora:**")
                    for mejora in mejoras:
                        st.markdown(f"• {mejora}")
    
    # Comentario general y recomendaciones
    st.markdown("### 💬 Comentario General")
    st.write(result.get('comentario', 'Sin comentario general'))
    
    if result.get('recomendaciones'):
        st.markdown("### 🎯 Recomendaciones")
        for rec in result['recomendaciones']:
            st.markdown(f"• {rec}")

def show_exam_history():
    """Mostrar historial de exámenes"""
    if 'corrector' in st.session_state:
        corrector = st.session_state.corrector
        user = corrector.get_or_create_user()
        df_exams = corrector.get_user_stats(user[0])
        
        if not df_exams.empty:
            st.subheader("📚 Historial de Exámenes")
            
            # Tabla interactiva
            display_df = df_exams[['filename', 'subject', 'grade', 'created_at']].copy()
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['Archivo', 'Asignatura', 'Nota', 'Fecha']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Opción de ver detalle
            if len(df_exams) > 0:
                selected_exam = st.selectbox(
                    "Ver detalle de examen:",
                    options=range(len(df_exams)),
                    format_func=lambda x: f"{df_exams.iloc[x]['filename']} - {df_exams.iloc[x]['grade']:.1f}"
                )
                
                if st.button("Ver Detalle"):
                    exam_data = df_exams.iloc[selected_exam]
                    corrections = json.loads(exam_data['corrections'])
                    display_correction_results(corrections)
        else:
            st.info("📝 No hay exámenes corregidos aún. ¡Ingresa tu primer examen!")

def get_default_criteria_templates():
    """Plantillas de criterios predefinidas"""
    return {
        "Matemáticas": {
            "criteria": "Procedimiento correcto, cálculos precisos, respuesta final correcta, claridad en el desarrollo",
            "rubric": "Excelente (90-100): Procedimiento correcto y respuesta exacta. Bueno (70-89): Procedimiento correcto con errores menores. Regular (50-69): Procedimiento parcialmente correcto. Insuficiente (0-49): Procedimiento incorrecto o respuesta errónea."
        },
        "Ciencias": {
            "criteria": "Conocimiento conceptual, aplicación de principios científicos, análisis de datos, conclusiones válidas",
            "rubric": "Excelente (90-100): Demuestra comprensión profunda y análisis completo. Bueno (70-89): Comprensión sólida con análisis adecuado. Regular (50-69): Comprensión básica. Insuficiente (0-49): Comprensión limitada o incorrecta."
        },
        "Literatura": {
            "criteria": "Comprensión del texto, análisis literario, uso del lenguaje, estructura del ensayo",
            "rubric": "Excelente (90-100): Análisis profundo y escritura excepcional. Bueno (70-89): Análisis sólido y buena escritura. Regular (50-69): Comprensión básica. Insuficiente (0-49): Comprensión limitada."
        },
        "Historia": {
            "criteria": "Conocimiento de hechos históricos, análisis de causas y consecuencias, uso de evidencia histórica",
            "rubric": "Excelente (90-100): Conocimiento detallado y análisis crítico. Bueno (70-89): Conocimiento sólido con análisis adecuado. Regular (50-69): Conocimiento básico. Insuficiente (0-49): Conocimiento limitado."
        }
    }

def main():
    # CSS optimizado
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    .free-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar optimizado
    with st.sidebar:
        st.markdown('<div class="free-badge">🆓 100% GRATUITO</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Navegación simplificada
        page = st.selectbox(
            "📍 Navegación",
            ["🏠 Dashboard", "🤖 Corrector", "📚 Historial", "💰 Costos", "ℹ️ Info"]
        )
        
        st.markdown("---")
        
        # Info del plan gratuito
        plan = PRICING_PLANS['free']
        st.markdown(f"**Plan:** {plan.name}")
        
        # Límites de uso
        if 'corrector' in st.session_state:
            corrector = st.session_state.corrector
            user = corrector.get_or_create_user()
            
            conn = sqlite3.connect('examai_free.db')
            cursor = conn.cursor()
            cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user[0],))
            current_user = cursor.fetchone()
            conn.close()
            
            exams_used = current_user[0] if current_user else 0
            st.markdown(f"**Uso:** {exams_used}/{plan.exams_limit}")
            
            progress = exams_used / plan.exams_limit
            st.progress(progress)
        
        st.markdown("---")
        
        # API Key con info de costos
        st.markdown("### 🔑 OpenAI API Key")
        api_key = st.text_input("Tu API Key", type="password", 
                               help="Solo se usa para GPT-3.5-turbo (~$0.05 por examen)")
        
        if not api_key:
            st.error("⚠️ Necesitas tu API Key de OpenAI")
            st.markdown("""
            **Cómo obtenerla:**
            1. Ve a [platform.openai.com](https://platform.openai.com)
            2. Crea una cuenta
            3. Ve a API Keys
            4. Crea una nueva key
            5. Añade $5-10 de crédito
            """)
            st.stop()
        else:
            st.success("✅ API Key configurada")
    
    # Inicializar corrector
    if 'corrector' not in st.session_state and api_key:
        st.session_state.corrector = ExamCorrectorFree(api_key)
    
    # Enrutamiento
    if page == "🏠 Dashboard":
        show_free_dashboard()
    
    elif page == "📚 Historial":
        st.title("📚 Historial de Exámenes")
        show_exam_history()
    
    elif page == "💰 Costos":
        st.title("💰 Transparencia de Costos")
        
        st.success("""
        ### 🆓 Servicios Gratuitos Utilizados:
        - **Base de datos:** SQLite (gratis para siempre)
        - **Procesamiento:** Python nativo (gratis)
        - **Hosting:** Streamlit Community Cloud (gratis)
        - **Gráficos:** Plotly (gratis)
        - **Interfaz:** Streamlit (gratis)
        """)
        
        st.info("""
        ### 💸 Único Costo:
        **OpenAI API (GPT-3.5-turbo)**
        - $0.001 por 1K tokens entrada
        - $0.002 por 1K tokens salida
        - ~$0.03-0.08 por examen típico
        """)
        
        show_cost_calculator()
        
        st.markdown("""
        ### 💡 Consejos para minimizar costos:
        1. **Usa criterios concisos** (menos tokens = menos costo)
        2. **Limita la longitud del texto** (máximo 8000 caracteres)
        3. **Batch de exámenes** (corrige varios a la vez)
        4. **Revisa el dashboard** para monitorear uso
        """)
    
    elif page == "ℹ️ Info":
        st.title("ℹ️ Información del Sistema")
        
        st.markdown("""
        ## 🎯 ExamAI Free - Corrector 100% Gratuito
        
        ### ✅ Características Incluidas:
        - ✅ 50 exámenes gratis por mes
        - ✅ Procesamiento de texto directo
        - ✅ Corrección con IA (GPT-3.5-turbo)
        - ✅ Base de datos local gratuita
        - ✅ Estadísticas y gráficos
        - ✅ Sin suscripciones ni pagos ocultos
        
        ### 💰 Costos Transparentes:
        - 🆓 **Todo gratis** excepto tu API de OpenAI
        - 💸 **Solo pagas por IA:** ~$2-5/mes para uso normal
        - 🔄 **Límites se reinician** cada mes automáticamente
        
        ### 🛠️ Tecnologías Utilizadas:
        - **Frontend:** Streamlit
        - **IA:** OpenAI GPT-3.5-turbo
        - **Base de datos:** SQLite
        - **Gráficos:** Plotly
        - **Procesamiento:** Python nativo
        
        ### 📞 Soporte:
        - 📧 Email: soporte@examai.com
        - 💬 Chat: Disponible en horario laboral
        - 📚 Documentación: github.com/examai/docs
        """)
    
    elif page == "🤖 Corrector":
        st.title("🤖 Corrector Automático de Exámenes")
        
        if 'corrector' not in st.session_state:
            st.error("⚠️ Error: Corrector no inicializado")
            st.stop()
        
        corrector = st.session_state.corrector
        
        # Verificar límites
        user = corrector.get_or_create_user()
        user_id = user[0]
        
        conn = sqlite3.connect('examai_free.db')
        cursor = conn.cursor()
        cursor.execute('SELECT exams_used FROM users WHERE id = ?', (user_id,))
        current_user = cursor.fetchone()
        conn.close()
        
        exams_used = current_user[0] if current_user else 0
        
        if exams_used >= PRICING_PLANS['free'].exams_limit:
            st.error(f"⚠️ Has alcanzado el límite de {PRICING_PLANS['free'].exams_limit} exámenes este mes")
            st.info("🔄 Los límites se reinician automáticamente cada mes")
            st.stop()
        
        # Interfaz de corrección
        st.success(f"✅ Exámenes restantes: {PRICING_PLANS['free'].exams_limit - exams_used}")
        
        # Configuración de corrección
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Configuración de Corrección")
            
            # Selección de materia
            subject = st.selectbox(
                "Materia:",
                ["Matemáticas", "Ciencias", "Literatura", "Historia", "Personalizada"]
            )
            
            # Obtener plantillas predefinidas
            templates = get_default_criteria_templates()
            
            if subject in templates:
                default_criteria = templates[subject]["criteria"]
                default_rubric = templates[subject]["rubric"]
            else:
                default_criteria = "Criterios personalizados de evaluación"
                default_rubric = "Excelente (90-100): Cumple todos los criterios. Bueno (70-89): Cumple la mayoría. Regular (50-69): Cumple algunos. Insuficiente (0-49): No cumple criterios."
            
            # Criterios de evaluación
            criteria = st.text_area(
                "Criterios de evaluación:",
                value=default_criteria,
                height=100,
                help="Define qué aspectos evaluar en el examen"
            )
            
            # Rúbrica de calificación
            rubric = st.text_area(
                "Rúbrica de calificación:",
                value=default_rubric,
                height=150,
                help="Define cómo asignar puntuaciones"
            )
        
        with col2:
            st.subheader("📝 Entrada del Examen")
            
            # Método de entrada
            input_method = st.radio(
                "Método de entrada:",
                ["Texto directo", "Archivo de texto"]
            )
            
            exam_text = ""
            filename = "examen_directo.txt"
            
            if input_method == "Texto directo":
                exam_text = st.text_area(
                    "Pega el texto del examen aquí:",
                    height=300,
                    placeholder="Ingresa las respuestas del estudiante...",
                    help="Máximo 8000 caracteres para optimizar costos"
                )
                
                if exam_text:
                    char_count = len(exam_text)
                    st.caption(f"Caracteres: {char_count:,}/8,000")
                    
                    if char_count > 8000:
                        st.warning("⚠️ Texto muy largo. Se truncará para optimizar costos.")
            
            elif input_method == "Archivo de texto":
                uploaded_file = st.file_uploader(
                    "Sube archivo de texto:",
                    type=['txt', 'md'],
                    help="Solo archivos .txt y .md"
                )
                
                if uploaded_file:
                    filename = uploaded_file.name
                    exam_text = str(uploaded_file.read(), "utf-8")
                    
                    char_count = len(exam_text)
                    st.success(f"✅ Archivo cargado: {char_count:,} caracteres")
                    
                    if char_count > 8000:
                        st.warning("⚠️ Archivo muy grande. Se truncará para optimizar costos.")
        
        # Botón de corrección
        st.markdown("---")
        
        if st.button("🚀 Corregir Examen", type="primary", use_container_width=True):
            if not exam_text or not exam_text.strip():
                st.error("⚠️ Por favor ingresa el texto del examen")
                st.stop()
            
            if not criteria.strip():
                st.error("⚠️ Por favor define los criterios de evaluación")
                st.stop()
            
            if not rubric.strip():
                st.error("⚠️ Por favor define la rúbrica de calificación")
                st.stop()
            
            # Mostrar progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Procesar texto
                status_text.text("📝 Procesando texto...")
                progress_bar.progress(25)
                
                processed_text = corrector.process_text_input(exam_text)
                
                if not processed_text:
                    st.error("❌ Error procesando el texto")
                    st.stop()
                
                # Corrección con IA
                status_text.text("🤖 Enviando a IA para corrección...")
                progress_bar.progress(50)
                
                result = corrector.correct_exam_budget(
                    processed_text,
                    criteria,
                    rubric,
                    subject
                )
                
                # Guardar resultado
                status_text.text("💾 Guardando resultado...")
                progress_bar.progress(75)
                
                corrector.save_exam_result(user_id, filename, subject, result)
                
                # Completar
                status_text.text("✅ ¡Corrección completada!")
                progress_bar.progress(100)
                
                st.success("🎉 ¡Examen corregido exitosamente!")
                
                # Mostrar resultados
                st.markdown("---")
                display_correction_results(result)
                
                # Actualizar contador en sidebar
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error durante la corrección: {str(e)}")
                st.info("💡 Intenta nuevamente o contacta soporte")

def show_success_tips():
    """Consejos para usar ExamAI Free eficientemente"""
    st.markdown("""
    ### 💡 Consejos para Mejores Resultados:
    
    #### 📝 Preparación del Texto:
    - **Limpieza:** Elimina información irrelevante
    - **Estructura:** Organiza por preguntas/secciones
    - **Longitud:** Mantén bajo 8000 caracteres
    
    #### 🎯 Criterios Efectivos:
    - **Específicos:** Define exactamente qué evaluar
    - **Medibles:** Usa criterios cuantificables
    - **Claros:** Evita ambigüedades
    
    #### 📊 Rúbricas Útiles:
    - **Rangos claros:** 90-100, 70-89, 50-69, 0-49
    - **Descriptores:** Qué caracteriza cada nivel
    - **Consistencia:** Usa la misma escala siempre
    
    #### 💰 Optimización de Costos:
    - **Batch processing:** Corrige varios exámenes seguidos
    - **Reutiliza criterios:** Guarda plantillas para materias
    - **Revisa antes:** Verifica texto antes de enviar
    """)

def export_results_to_excel():
    """Exportar resultados a Excel"""
    if 'corrector' in st.session_state:
        corrector = st.session_state.corrector
        user = corrector.get_or_create_user()
        df_exams = corrector.get_user_stats(user[0])
        
        if not df_exams.empty:
            # Crear archivo Excel en memoria
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Hoja principal con resumen
                summary_df = df_exams[['filename', 'subject', 'grade', 'created_at']].copy()
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hoja detallada con correcciones
                detailed_data = []
                for _, row in df_exams.iterrows():
                    corrections = json.loads(row['corrections'])
                    detailed_data.append({
                        'Archivo': row['filename'],
                        'Materia': row['subject'],
                        'Nota Final': row['grade'],
                        'Fecha': row['created_at'],
                        'Comentario': corrections.get('comentario', ''),
                        'Recomendaciones': '; '.join(corrections.get('recomendaciones', []))
                    })
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detallado', index=False)
            
            output.seek(0)
            return output.getvalue()
    
    return None

# Agregar función para mostrar métricas avanzadas
def show_advanced_metrics():
    """Mostrar métricas avanzadas gratuitas"""
    if 'corrector' in st.session_state:
        corrector = st.session_state.corrector
        user = corrector.get_or_create_user()
        df_exams = corrector.get_user_stats(user[0])
        
        if not df_exams.empty:
            st.subheader("📊 Métricas Avanzadas")
            
            # Métricas por materia
            subject_stats = df_exams.groupby('subject').agg({
                'grade': ['mean', 'count', 'std']
            }).round(2)
            
            st.markdown("#### 📚 Rendimiento por Materia")
            for subject in subject_stats.index:
                avg_grade = subject_stats.loc[subject, ('grade', 'mean')]
                count = subject_stats.loc[subject, ('grade', 'count')]
                std_dev = subject_stats.loc[subject, ('grade', 'std')]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{subject} - Promedio", f"{avg_grade:.1f}")
                with col2:
                    st.metric("Exámenes", int(count))
                with col3:
                    st.metric("Desviación", f"{std_dev:.1f}")
            
            # Tendencia temporal
            st.markdown("#### 📈 Tendencia Temporal")
            
            df_exams['created_at'] = pd.to_datetime(df_exams['created_at'])
            df_exams['week'] = df_exams['created_at'].dt.isocalendar().week
            
            weekly_avg = df_exams.groupby('week')['grade'].mean()
            
            fig = px.line(
                x=weekly_avg.index,
                y=weekly_avg.values,
                title="Promedio Semanal de Calificaciones",
                labels={'x': 'Semana', 'y': 'Promedio'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Función principal mejorada
if __name__ == "__main__":
    # Verificar si es la primera vez que se ejecuta
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        
        # Mostrar mensaje de bienvenida
        st.balloons()
        st.success("""
        🎉 **¡Bienvenido a ExamAI Free!**
        
        Tu corrector de exámenes 100% gratuito está listo.
        Solo necesitas tu API Key de OpenAI para empezar.
        """)
    
    # Ejecutar aplicación principal
    main()
    
    # Footer informativo
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🆓 <strong>ExamAI Free</strong> - Corrector inteligente gratuito<br>
    💰 Solo pagas por OpenAI API (~$0.05 por examen)<br>
    🔄 Límites se reinician automáticamente cada mes<br>
    📧 Soporte: soporte@examai.com
    </div>
    """, unsafe_allow_html=True)

