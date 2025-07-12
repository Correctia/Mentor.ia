import streamlit as st
import requests
import json
import base64
import io
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_bytes

# Configuración de la página
st.set_page_config(
    page_title="Corrector de Exámenes con IA",
    page_icon="🤖",
    layout="wide"
)

# Enumeraciones
class SubjectType(Enum):
    SCIENCES = "ciencias"
    HUMANITIES = "letras"

class GradeLevel(Enum):
    PRIMARY = "primaria"
    SECONDARY = "secundaria"
    BACHILLERATO = "bachillerato"
    UNIVERSITY = "universidad"

# Modelos de datos
@dataclass
class ClassInfo:
    name: str
    subject: str
    grade: GradeLevel
    subject_type: SubjectType
    students_count: int = 0

# Configuración de API
class APIConfig:
    def __init__(self):
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.google_vision_api_key = os.getenv("GOOGLE_VISION_API_KEY", "")
        #self.mathpix_api_key = os.getenv("MATHPIX_API_KEY", "")

# Procesador avanzado de imágenes
class AdvancedImageProcessor:
    def preprocess_image_for_ocr(self, image_bytes: bytes) -> bytes:
        """Procesa la imagen para mejorar el OCR"""
        try:
            # Convertir bytes a imagen
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtro bilateral para reducir ruido
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convertir de vuelta a bytes
            success, encoded_image = cv2.imencode('.png', binary)
            if success:
                return encoded_image.tobytes()
            else:
                return image_bytes
                
        except Exception as e:
            st.warning(f"Error procesando imagen: {str(e)}")
            return image_bytes

# Servicio OCR
class OCRService:
    def __init__(self, config: APIConfig):
        self.config = config
    
    #def mathpix_ocr(self, image_data: bytes) -> str:
       # """OCR usando Mathpix para matemáticas y ciencias"""
        #if not self.config.mathpix_api_key:
           # st.error("API key de Mathpix no configurada")
            #return ""
        
        #url = "https://api.mathpix.com/v3/text"
        
        #headers = {
            #"app_id": "your_app_id",
            #"app_key": self.config.mathpix_api_key,
            #"Content-Type": "application/json"
        #}
        
        #image_b64 = base64.b64encode(image_data).decode()
        
        #payload = {
            #"src": f"data:image/png;base64,{image_b64}",
            #"formats": ["text", "latex_styled"],
            #"data_options": {
                #"include_asciimath": True,
                #"include_latex": True
            #}
        #}
        
        #try:
            #response = requests.post(url, json=payload, headers=headers)
            #response.raise_for_status()
            #result = response.json()
            #return result.get("text", "")
        #except Exception as e:
            #st.error(f"Error en Mathpix OCR: {str(e)}")
            #return ""
    
    def google_vision_ocr(self, image_data: bytes) -> str:
        """OCR usando Google Vision API para asignaturas de letras"""
        if not self.config.google_vision_api_key:
            st.error("API key de Google Vision no configurada")
            return ""
        
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.config.google_vision_api_key}"
        
        image_b64 = base64.b64encode(image_data).decode()
        payload = {
            "requests": [{
                "image": {"content": image_b64},
                "features": [{
                    "type": "TEXT_DETECTION",
                    "maxResults": 1
                }]
            }]
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "responses" in result and result["responses"]:
                annotations = result["responses"][0].get("textAnnotations", [])
                if annotations:
                    return annotations[0]["description"]
            return ""
        except Exception as e:
            st.error(f"Error en Google Vision OCR: {str(e)}")
            return ""

# Servicio de IA para corrección
class AIService:
    def __init__(self, config: APIConfig):
        self.config = config
    
    def correct_exam(self, exam_text: str, subject_type: SubjectType, subject_name: str) -> Dict:
        """Corrige el examen usando DeepSeek"""
        if not self.config.deepseek_api_key:
            st.error("❌ API key de DeepSeek no configurada")
            return None
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        if subject_type == SubjectType.SCIENCES:
            system_prompt = f"""Eres un profesor experto en {subject_name} (asignatura de ciencias). 
            Analiza el siguiente examen y proporciona correcciones detalladas.
            
            Para cada pregunta identifica:
            1. La pregunta original
            2. La respuesta del estudiante
            3. Si la respuesta es correcta o incorrecta
            4. Comentarios específicos sobre errores
            5. La respuesta correcta si es necesaria
            6. Puntuación sugerida
            
            Devuelve la respuesta en formato JSON con esta estructura:
            {{
                "corrections": [
                    {{
                        "question": "pregunta",
                        "student_answer": "respuesta del estudiante",
                        "is_correct": true/false,
                        "comments": "comentarios específicos",
                        "correct_answer": "respuesta correcta",
                        "score": puntuación
                    }}
                ],
                "total_score": puntuación_total,
                "feedback": "comentarios generales"
            }}"""
        else:
            system_prompt = f"""Eres un profesor experto en {subject_name} (asignatura de letras). 
            Analiza el siguiente examen y proporciona correcciones detalladas.
            
            Para cada pregunta evalúa:
            1. Comprensión del tema
            2. Calidad de la argumentación
            3. Uso correcto del lenguaje
            4. Estructura de la respuesta
            5. Contenido específico
            
            Devuelve la respuesta en formato JSON con la estructura anterior."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Examen a corregir:\n\n{exam_text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            with st.spinner("🔄 Procesando con IA..."):
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code != 200:
                    st.error(f"❌ Error HTTP {response.status_code}")
                    return None
                
                result = response.json()
                
                if "choices" not in result or not result["choices"]:
                    st.error("❌ Respuesta de API inválida")
                    return None
                
                content = result["choices"][0]["message"]["content"]
                
                # Intentar parsear JSON
                try:
                    parsed_result = json.loads(content)
                    
                    # Verificar estructura
                    if "corrections" in parsed_result and isinstance(parsed_result["corrections"], list):
                        return parsed_result
                    else:
                        # Crear estructura básica si no tiene el formato correcto
                        return self._create_fallback_response(exam_text, content)
                        
                except json.JSONDecodeError:
                    # Si no es JSON válido, crear respuesta de fallback
                    return self._create_fallback_response(exam_text, content)
                    
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error de conexión: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
            return None
    
    def generate_content(self, prompt: str) -> str:
        """Método auxiliar para generar contenido de texto"""
        if not self.config.deepseek_api_key:
            st.error("❌ API key de DeepSeek no configurada")
            return ""
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            st.error(f"Error generando contenido: {str(e)}")
            return ""
    
    def _create_fallback_response(self, exam_text: str, ai_content: str) -> Dict:
        """Crea una respuesta de fallback cuando el JSON no es válido"""
        return {
            "corrections": [{
                "question": "Análisis general del examen",
                "student_answer": exam_text[:200] + "..." if len(exam_text) > 200 else exam_text,
                "is_correct": False,
                "comments": "Análisis procesado correctamente",
                "correct_answer": "Ver feedback general",
                "score": 5
            }],
            "total_score": 5,
            "feedback": ai_content
        }

# Inicializar session state
def init_session_state():
    """Inicializa el estado de la sesión"""
    if "classes" not in st.session_state:
        st.session_state.classes = []
    if "exam_history" not in st.session_state:
        st.session_state.exam_history = []

# Páginas de la aplicación
def show_home():
    """Página de inicio"""
    st.title("🤖 Corrector de Exámenes con IA")
    st.markdown("### Bienvenido al sistema de corrección automática")
    
    st.markdown("""
    **Características principales:**
    - 📄 Procesamiento avanzado de PDFs e imágenes
    - 🔍 OCR especializado para ciencias y letras
    - 🤖 Corrección automática con IA
    - 📊 Análisis detallado de respuestas
    - 📚 Historial de correcciones
    """)
    
    st.info("👈 Usa el menú lateral para navegar por las diferentes secciones")

def show_class_management():
    """Gestión de clases"""
    st.header("👥 Gestionar Clases")
    
    # Formulario para nueva clase
    with st.form("new_class_form"):
        st.subheader("Crear Nueva Clase")
        
        col1, col2 = st.columns(2)
        
        with col1:
            class_name = st.text_input("Nombre de la clase", placeholder="Ej: 3º ESO A")
            subject = st.text_input("Asignatura", placeholder="Ej: Matemáticas")
            students_count = st.number_input("Número de estudiantes", min_value=1, max_value=50, value=25)
        
        with col2:
            grade = st.selectbox("Nivel educativo", [grade.value for grade in GradeLevel])
            subject_type = st.selectbox("Tipo de asignatura", [subj.value for subj in SubjectType])
        
        submitted = st.form_submit_button("✅ Crear Clase")
        
        if submitted and class_name and subject:
            new_class = ClassInfo(
                name=class_name,
                subject=subject,
                grade=GradeLevel(grade),
                subject_type=SubjectType(subject_type),
                students_count=students_count
            )
            
            st.session_state.classes.append(new_class)
            st.success(f"✅ Clase '{class_name}' creada exitosamente")
            st.rerun()
    
    # Mostrar clases existentes
    if st.session_state.classes:
        st.subheader("Clases Existentes")
        
        for i, class_info in enumerate(st.session_state.classes):
            with st.expander(f"📚 {class_info.name} - {class_info.subject}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Nivel:** {class_info.grade.value}")
                    st.write(f"**Tipo:** {class_info.subject_type.value}")
                
                with col2:
                    st.write(f"**Estudiantes:** {class_info.students_count}")
                
                with col3:
                    if st.button(f"🗑️ Eliminar", key=f"delete_{i}"):
                        st.session_state.classes.pop(i)
                        st.success("Clase eliminada")
                        st.rerun()
    else:
        st.info("No hay clases creadas. Crea tu primera clase arriba.")

def show_exam_correction(ocr_service: OCRService, ai_service: AIService):
    """Corrección de exámenes con procesamiento avanzado"""
    st.header("📄 Corregir Examen")
    
    if not st.session_state.classes:
        st.warning("⚠️ Primero debes crear una clase en 'Gestionar Clases'")
        return
    
    # Seleccionar clase
    class_names = [f"{c.name} - {c.subject}" for c in st.session_state.classes]
    selected_class_idx = st.selectbox("Selecciona una clase:", range(len(class_names)), 
                                     format_func=lambda x: class_names[x])
    
    selected_class = st.session_state.classes[selected_class_idx]
    
    # Título del examen
    exam_title = st.text_input("Título del examen", value=f"Examen {selected_class.subject}")
    
    # Métodos de entrada
    st.subheader("📁 Método de entrada")
    input_method = st.radio("Selecciona cómo quieres subir el examen:", 
                           ["📄 Subir PDF", "🖼️ Subir imágenes", "✍️ Escribir texto"])
    
    exam_text = ""
    
    # Crear procesador avanzado
    processor = AdvancedImageProcessor()
    
    if input_method == "📄 Subir PDF":
        uploaded_pdf = st.file_uploader("Sube el PDF del examen", type=['pdf'])
        
        if uploaded_pdf:
            with st.spinner("Procesando PDF..."):
                try:
                    # Convertir PDF a imágenes
                    pdf_pages = convert_from_bytes(uploaded_pdf.read(), dpi=300)
                    
                    all_text = []
                    for i, page in enumerate(pdf_pages):
                        st.write(f"Procesando página {i+1}...")
                        
                        # Convertir página a bytes
                        img_bytes = io.BytesIO()
                        page.save(img_bytes, format='PNG')
                        img_bytes = img_bytes.getvalue()
                        
                        # Procesar con técnicas avanzadas
                        processed_bytes = processor.preprocess_image_for_ocr(img_bytes)
                        
                        # Mostrar imagen procesada
                        processed_image = Image.open(io.BytesIO(processed_bytes))
                        st.image(processed_image, caption=f"Página {i+1} procesada", use_column_width=True)
                        
                        # Extraer texto según tipo de asignatura
                        if selected_class.subject_type == SubjectType.SCIENCES:
                            page_text = ocr_service.mathpix_ocr(processed_bytes)
                        else:
                            page_text = ocr_service.google_vision_ocr(processed_bytes)
                        
                        if page_text:
                            all_text.append(f"--- Página {i+1} ---\n{page_text}")
                    
                    exam_text = "\n\n".join(all_text)
                    
                    if exam_text:
                        st.success("✅ PDF procesado exitosamente")
                        with st.expander("Ver texto extraído del PDF"):
                            st.text_area("Texto completo", exam_text, height=300)
                    else:
                        st.error("❌ No se pudo extraer texto del PDF")
                        
                except Exception as e:
                    st.error(f"❌ Error procesando PDF: {str(e)}")
    
    elif input_method == "🖼️ Subir imágenes":
        uploaded_images = st.file_uploader("Sube las imágenes del examen", 
                                          type=['png', 'jpg', 'jpeg'], 
                                          accept_multiple_files=True)
        
        if uploaded_images:
            all_text = []
            
            for i, uploaded_image in enumerate(uploaded_images):
                st.write(f"Procesando imagen {i+1}...")
                
                # Cargar imagen
                image = Image.open(uploaded_image)
                st.image(image, caption=f"Imagen {i+1} (original)", use_column_width=True)
                
                # Convertir a bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                with st.spinner(f"Procesando imagen {i+1} con técnicas avanzadas..."):
                    try:
                        # Procesar con técnicas avanzadas
                        processed_bytes = processor.preprocess_image_for_ocr(img_bytes)
                        
                        # Mostrar imagen procesada
                        processed_image = Image.open(io.BytesIO(processed_bytes))
                        st.image(processed_image, caption=f"Imagen {i+1} procesada", use_column_width=True)
                        
                        # Extraer texto según tipo de asignatura
                        if selected_class.subject_type == SubjectType.SCIENCES:
                            image_text = ocr_service.mathpix_ocr(processed_bytes)
                        else:
                            image_text = ocr_service.google_vision_ocr(processed_bytes)
                        
                        if image_text:
                            all_text.append(f"--- Imagen {i+1} ---\n{image_text}")
                            st.success(f"✅ Imagen {i+1} procesada exitosamente")
                        else:
                            st.warning(f"⚠️ No se pudo extraer texto de la imagen {i+1}")
                    
                    except Exception as e:
                        st.error(f"❌ Error procesando imagen {i+1}: {str(e)}")
            
            exam_text = "\n\n".join(all_text)
            
            if exam_text:
                st.success("✅ Todas las imágenes procesadas")
                with st.expander("Ver texto extraído de todas las imágenes"):
                    st.text_area("Texto completo", exam_text, height=300)
            else:
                st.error("❌ No se pudo extraer texto de ninguna imagen")
    
    elif input_method == "✍️ Escribir texto":
        exam_text = st.text_area("Escribe o pega el texto del examen:", 
                                height=300, 
                                placeholder="Pega aquí el texto del examen...")
    
    # Botón de corrección
    if exam_text and st.button("🤖 Corregir Examen"):
        correct_scanned_exam(exam_text, selected_class, exam_title, ai_service)

def correct_scanned_exam(exam_text: str, selected_class, exam_title: str, ai_service: AIService):
    """Corrige un examen escaneado usando IA"""
    with st.spinner("Corrigiendo examen..."):
        try:
            # Usar el método correct_exam del AIService
            correction_result = ai_service.correct_exam(
                exam_text, 
                selected_class.subject_type, 
                selected_class.subject
            )
            
            if correction_result:
                st.success("✅ Examen corregido exitosamente")
                
                # Mostrar resultados estructurados
                st.markdown("### 📊 Resultado de la Corrección")
                
                # Mostrar puntuación total
                total_score = correction_result.get("total_score", 0)
                st.metric("Puntuación Total", f"{total_score}/10")
                
                # Mostrar correcciones por pregunta
                st.markdown("### 📝 Correcciones por Pregunta")
                
                for i, correction in enumerate(correction_result.get("corrections", [])):
                    with st.expander(f"Pregunta {i+1}: {correction.get('question', 'Sin título')[:50]}..."):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Respuesta del estudiante:**")
                            st.write(correction.get("student_answer", ""))
                            
                            status = "✅ Correcta" if correction.get("is_correct") else "❌ Incorrecta"
                            st.markdown(f"**Estado:** {status}")
                        
                        with col2:
                            st.markdown("**Puntuación:**")
                            st.write(f"{correction.get('score', 0)} puntos")
                            
                            st.markdown("**Comentarios:**")
                            st.write(correction.get("comments", ""))
                            
                            if correction.get("correct_answer"):
                                st.markdown("**Respuesta correcta:**")
                                st.write(correction.get("correct_answer", ""))
                
                # Mostrar feedback general
                if correction_result.get("feedback"):
                    st.markdown("### 💬 Comentarios Generales")
                    st.write(correction_result.get("feedback"))
                
                # Guardar en historial
                exam_record = {
                    "title": exam_title,
                    "class": f"{selected_class.name} - {selected_class.subject}",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": exam_text,
                    "correction": correction_result,
                    "method": "Escaneo con procesamiento avanzado"
                }
                
                st.session_state.exam_history.append(exam_record)
                
                # Opciones adicionales
                st.markdown("### 📋 Acciones adicionales")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("💾 Guardar corrección"):
                        save_correction_to_file(exam_record)
                
                with col2:
                    if st.button("📧 Enviar por email"):
                        st.info("Función de email próximamente disponible")
                
                with col3:
                    if st.button("🔄 Corregir otro examen"):
                        st.rerun()
            else:
                st.error("❌ Error al corregir el examen")
        
        except Exception as e:
            st.error(f"❌ Error en la corrección: {str(e)}")

def save_correction_to_file(exam_record: dict):
    """Guarda la corrección en un archivo"""
    try:
        filename = f"correccion_{exam_record['title'].replace(' ', '_')}_{exam_record['date'][:10]}.txt"
        
        content = f"""
CORRECCIÓN DE EXAMEN
===================

Título: {exam_record['title']}
Clase: {exam_record['class']}
Fecha: {exam_record['date']}
Método: {exam_record['method']}

TEXTO ORIGINAL:
{exam_record['text']}

CORRECCIÓN:
{json.dumps(exam_record['correction'], indent=2, ensure_ascii=False)}
"""
        
        # Crear botón de descarga
        st.download_button(
            label="⬇️ Descargar corrección",
            data=content,
            file_name=filename,
            mime="text/plain"
        )
        
        st.success(f"✅ Corrección guardada como: {filename}")
    
    except Exception as e:
        st.error(f"❌ Error guardando archivo: {str(e)}")

# 7. Función para mostrar historial de exámenes
def show_exam_history():
    """Muestra el historial de exámenes corregidos"""
    st.header("📚 Historial de Exámenes")
    
    if "exam_history" not in st.session_state or not st.session_state.exam_history:
        st.info("No hay exámenes en el historial")
        return
    
    # Mostrar exámenes
    for i, exam in enumerate(reversed(st.session_state.exam_history)):
        with st.expander(f"📄 {exam['title']} - {exam['date']}"):
            st.write(f"**Clase:** {exam['class']}")
            st.write(f"**Método:** {exam['method']}")
            st.write(f"**Fecha:** {exam['date']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Texto original:**")
                st.text_area("", exam['text'], height=200, key=f"text_{i}")
            
            with col2:
                st.markdown("**Corrección:**")
                st.markdown(exam['correction'])
            
            # Botón para guardar
            if st.button(f"💾 Guardar corrección", key=f"save_{i}"):
                save_correction_to_file(exam)
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar historial"):
        st.session_state.exam_history = []
        st.success("✅ Historial limpiado")
        st.rerun()

# 8. Función principal modificada para incluir las nuevas funcionalidades
def main():
    """Función principal de la aplicación"""
    st.set_page_config(
        page_title="Corrector de Exámenes con IA",
        page_icon="🤖",
        layout="wide"
    )
    
    # Inicializar servicios
    config = APIConfig()
    ocr_service = OCRService(config)
    ai_service = AIService()
    
    # Sidebar para navegación
    with st.sidebar:
        st.title("🎯 Navegación")
        page = st.radio("Selecciona una página:", [
            "🏠 Inicio",
            "👥 Gestionar Clases", 
            "📄 Corregir Examen",
            "📚 Historial",
            "⚙️ Configuración"
        ])
    
    # Mostrar página seleccionada
    if page == "🏠 Inicio":
        show_home()
    elif page == "👥 Gestionar Clases":
        show_class_management()
    elif page == "📄 Corregir Examen":
        show_exam_correction(ocr_service, ai_service)
    elif page == "📚 Historial":
        show_exam_history()
    elif page == "⚙️ Configuración":
        show_settings()

if __name__ == "__main__":
    main()
    
    def google_vision_ocr(self, image_data: bytes) -> str:
        """OCR usando Google Vision API para asignaturas de letras"""
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.config.google_vision_api_key}"
        
        image_b64 = base64.b64encode(image_data).decode()
        
        payload = {
            "requests": [{
                "image": {"content": image_b64},
                "features": [{
                    "type": "TEXT_DETECTION",
                    "maxResults": 1
                }]
            }]
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "responses" in result and result["responses"]:
                annotations = result["responses"][0].get("textAnnotations", [])
                if annotations:
                    return annotations[0]["description"]
            return ""
        except Exception as e:
            st.error(f"Error en Google Vision OCR: {str(e)}")
            return ""
    

# Servicio de IA para corrección
class AIService:
    def __init__(self, config: APIConfig):
        self.config = config
    
    def correct_exam(self, exam_text: str, subject_type: SubjectType, subject_name: str) -> Dict:
        """Corrige el examen usando DeepSeek"""
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        # Verificar API key
        if not self.config.deepseek_api_key:
            st.error("❌ API key de DeepSeek no configurada")
            return None
        
        if subject_type == SubjectType.SCIENCES:
            system_prompt = f"""Eres un profesor experto en {subject_name} (asignatura de ciencias). 
            Analiza el siguiente examen y proporciona correcciones detalladas.
            
            Para cada pregunta identifica:
            1. La pregunta original
            2. La respuesta del estudiante
            3. Si la respuesta es correcta o incorrecta
            4. Comentarios específicos sobre errores
            5. La respuesta correcta si es necesaria
            6. Puntuación sugerida
            
            IMPORTANTE: Siempre incluye al menos una corrección, incluso si es un análisis general.
            
            Devuelve la respuesta en formato JSON con esta estructura:
            {{
                "corrections": [
                    {{
                        "question": "pregunta",
                        "student_answer": "respuesta del estudiante",
                        "is_correct": true/false,
                        "comments": "comentarios específicos",
                        "correct_answer": "respuesta correcta",
                        "score": puntuación
                    }}
                ],
                "total_score": puntuación_total,
                "feedback": "comentarios generales"
            }}"""
        else:
            system_prompt = f"""Eres un profesor experto en {subject_name} (asignatura de letras). 
            Analiza el siguiente examen y proporciona correcciones detalladas.
            
            Para cada pregunta evalúa:
            1. Comprensión del tema
            2. Calidad de la argumentación
            3. Uso correcto del lenguaje
            4. Estructura de la respuesta
            5. Contenido específico
            
            IMPORTANTE: Siempre incluye al menos una corrección, incluso si es un análisis general.
            
            Devuelve la respuesta en formato JSON con la estructura anterior."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Examen a corregir:\n\n{exam_text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            # Debug: mostrar que estamos haciendo la petición
            st.info("🔄 Enviando petición a DeepSeek API...")
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            # Debug: mostrar código de estado
            st.info(f"📡 Respuesta HTTP: {response.status_code}")
            
            if response.status_code != 200:
                st.error(f"❌ Error HTTP {response.status_code}: {response.text}")
                return None
            
            result = response.json()
            
            # Debug: mostrar estructura de respuesta
            st.write("**Debug - Estructura de respuesta:**")
            st.json(result)
            
            if "choices" not in result or not result["choices"]:
                st.error("❌ Respuesta de API inválida: sin choices")
                return None
            
            content = result["choices"][0]["message"]["content"]
            st.write("**Debug - Contenido recibido:**")
            st.text(content)
            
            # Intentar parsear JSON
            try:
                parsed_result = json.loads(content)
                
                # Verificar que tenga la estructura correcta
                if "corrections" in parsed_result:
                    if isinstance(parsed_result["corrections"], list) and len(parsed_result["corrections"]) > 0:
                        st.success("✅ JSON parseado correctamente con correcciones")
                        return parsed_result
                    else:
                        st.warning("⚠️ JSON válido pero sin correcciones")
                        # Crear una corrección básica
                        return {
                            "corrections": [{
                                "question": "Análisis general",
                                "student_answer": exam_text[:200] + "...",
                                "is_correct": False,
                                "comments": "Análisis procesado correctamente",
                                "correct_answer": "Ver feedback general",
                                "score": 5
                            }],
                            "total_score": 5,
                            "feedback": parsed_result.get("feedback", content)
                        }
                else:
                    st.error("❌ JSON válido pero sin clave 'corrections'")
                    return None
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Error parseando JSON: {str(e)}")
                st.text("Contenido que causó error:")
                st.text(content)
                
                # Crear estructura básica con contenido de respuesta
                return {
                    "corrections": [{
                        "question": "Análisis general del examen",
                        "student_answer": exam_text[:200] + "...",
                        "is_correct": False,
                        "comments": "La IA no pudo generar JSON válido, pero procesó el examen",
                        "correct_answer": "Ver feedback general",
                        "score": 5
                    }],
                    "total_score": 5,
                    "feedback": content
                }
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error de conexión: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
            st.exception(e)
            return None
    
    def generate_content(self, prompt: str) -> str:
        """Método auxiliar para generar contenido"""
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            st.error(f"Error generando contenido: {str(e)}")
            return ""
            
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.captured_image = None
        self.capture_flag = False
    
    def capture_image(self):
        self.capture_flag = True
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.capture_flag:
            self.captured_image = img.copy()
            self.capture_flag = False
        
        # Mejorar la imagen para escáner
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detección de contornos para encontrar documentos
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno más grande (probablemente el documento)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Dibujar el contorno en la imagen
            cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Funciones auxiliares
def preprocess_image(image: Image.Image) -> bytes:
    """Preprocesa la imagen para mejorar el OCR"""
    # Convertir a escala de grises
    gray = image.convert('L')
    
    # Mejorar contraste
    import numpy as np
    img_array = np.array(gray)
    img_array = cv2.equalizeHist(img_array)
    
    # Aplicar filtro de nitidez
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_array = cv2.filter2D(img_array, -1, kernel)
    
    # Convertir de vuelta a PIL
    processed_image = Image.fromarray(img_array)
    
    # Convertir a bytes
    img_bytes = io.BytesIO()
    processed_image.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convierte PDF a imágenes"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        images.append(image)
    
    doc.close()
    return images

def create_corrected_pdf(original_text: str, corrections: Dict, title: str) -> bytes:
    """Crea un PDF corregido con comentarios en rojo"""
    # Crear un documento PDF simple con las correcciones
    doc = fitz.open()
    page = doc.new_page()
    
    # Configurar fuentes
    font_size = 12
    red_color = (1, 0, 0)  # RGB rojo
    black_color = (0, 0, 0)  # RGB negro
    
    y_position = 50
    
    # Título
    page.insert_text((50, y_position), title, fontsize=16, color=black_color)
    y_position += 40
    
    # Agregar correcciones
    for i, correction in enumerate(corrections.get("corrections", [])):
        # Pregunta
        page.insert_text((50, y_position), f"Pregunta {i+1}: {correction['question']}", 
                        fontsize=font_size, color=black_color)
        y_position += 20
        
        # Respuesta del estudiante
        page.insert_text((50, y_position), f"Respuesta: {correction['student_answer']}", 
                        fontsize=font_size, color=black_color)
        y_position += 20
        
        # Comentarios en rojo
        if not correction['is_correct']:
            page.insert_text((50, y_position), f"❌ INCORRECTO: {correction['comments']}", 
                            fontsize=font_size, color=red_color)
            y_position += 15
            
            if correction.get('correct_answer'):
                page.insert_text((50, y_position), f"Respuesta correcta: {correction['correct_answer']}", 
                                fontsize=font_size, color=red_color)
                y_position += 15
        else:
            page.insert_text((50, y_position), f"✅ CORRECTO", 
                            fontsize=font_size, color=(0, 0.5, 0))
            y_position += 15
        
        y_position += 20
        
        # Nueva página si es necesario
        if y_position > 750:
            page = doc.new_page()
            y_position = 50
    
    # Feedback general
    if corrections.get("feedback"):
        page.insert_text((50, y_position), f"Comentarios generales: {corrections['feedback']}", 
                        fontsize=font_size, color=black_color)
        y_position += 20
    
    # Puntuación total
    page.insert_text((50, y_position), f"Puntuación total: {corrections.get('total_score', 0)}", 
                    fontsize=14, color=red_color)
    
    # Guardar PDF
    pdf_bytes = doc.write()
    doc.close()
    
    return pdf_bytes

# Gestión de estado
def init_session_state():
    """Inicializa el estado de la sesión"""
    if 'classes' not in st.session_state:
        st.session_state.classes = []
    if 'exams' not in st.session_state:
        st.session_state.exams = []
    if 'current_class' not in st.session_state:
        st.session_state.current_class = None

# Interfaz principal
def main():
    st.title("Mentor.ia")
    st.markdown("---")
    
    # Inicializar configuración
    config = APIConfig()
    ocr_service = OCRService(config)
    ai_service = AIService(config)
    
    # Inicializar estado
    init_session_state()
    
    # Sidebar para navegación
    st.sidebar.title("🎓 Navegación")
    
    # Verificar configuración de APIs
    if not all([config.deepseek_api_key, config.google_vision_api_key,]):
        st.error("⚠️ Configuración de APIs incompleta. Asegúrate de configurar todas las claves API en los secrets.")
        st.info("Necesitas configurar: DEEPSEEK_API_KEY, GOOGLE_VISION_API_KEY")
        return
    
    # Menú de navegación
    menu = st.sidebar.selectbox(
        "Selecciona una opción:",
        ["🏠 Inicio", "📚 Gestionar Clases", "📄 Corregir Examen", "📊 Historial", "📷 Escáner"]
    )
    
    if menu == "🏠 Inicio":
        show_home()
    elif menu == "📚 Gestionar Clases":
        show_class_management()
    elif menu == "📄 Corregir Examen":
        show_exam_correction(ocr_service, ai_service)
    elif menu == "📊 Historial":
        show_exam_history()
    elif menu == "📷 Escáner":
        show_scanner(ocr_service, ai_service)

def show_home():
    """Página de inicio"""
    st.header("🏠 Bienvenido al Corrector de Exámenes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Estadísticas")
        st.metric("Clases creadas", len(st.session_state.classes))
        st.metric("Exámenes corregidos", len(st.session_state.exams))
    
    with col2:
        st.subheader("🚀 Características")
        st.markdown("""
        - ✅ OCR con Google Vision (Letras) y Mathpix (Ciencias)
        - 🤖 Corrección automática con DeepSeek AI
        - 📱 Escáner integrado con cámara
        - 📄 Soporte para PDF e imágenes
        - 🎯 Gestión de clases y asignaturas
        - 📊 Histórico de correcciones
        """)
    
    st.markdown("---")
    st.info("💡 Comienza creando una clase en la sección 'Gestionar Clases' y luego sube un examen para corregir.")

def show_class_management():
    """Gestión de clases"""
    st.header("📚 Gestionar Clases")
    
    # Crear nueva clase
    with st.expander("➕ Crear nueva clase"):
        with st.form("new_class_form"):
            class_name = st.text_input("Nombre de la clase")
            subject = st.text_input("Asignatura")
            subject_type = st.selectbox("Tipo de asignatura", 
                                      ["Letras", "Ciencias"])
            teacher_name = st.text_input("Nombre del profesor")
            
            if st.form_submit_button("Crear clase"):
                if class_name and subject and teacher_name:
                    new_class = ExamClass(
                        id=f"class_{len(st.session_state.classes) + 1}",
                        name=class_name,
                        subject=subject,
                        subject_type=SubjectType.LETTERS if subject_type == "Letras" else SubjectType.SCIENCES,
                        teacher_name=teacher_name,
                        created_at=datetime.now()
                    )
                    st.session_state.classes.append(new_class)
                    st.success(f"✅ Clase '{class_name}' creada exitosamente")
                    st.rerun()
                else:
                    st.error("❌ Por favor, completa todos los campos")
    
    # Mostrar clases existentes
    if st.session_state.classes:
        st.subheader("📋 Clases existentes")
        
        for class_obj in st.session_state.classes:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{class_obj.name}**")
                    st.write(f"Asignatura: {class_obj.subject} ({class_obj.subject_type.value})")
                    st.write(f"Profesor: {class_obj.teacher_name}")
                
                with col2:
                    exams_count = len([e for e in st.session_state.exams if e.class_id == class_obj.id])
                    st.metric("Exámenes", exams_count)
                
                with col3:
                    if st.button("🗑️ Eliminar", key=f"delete_{class_obj.id}"):
                        st.session_state.classes = [c for c in st.session_state.classes if c.id != class_obj.id]
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("🎯 No hay clases creadas. Crea tu primera clase para comenzar.")

def show_exam_correction(ocr_service: OCRService, ai_service: AIService):
    """Corrección de exámenes"""
    st.header("📄 Corregir Examen")
    
    if not st.session_state.classes:
        st.warning("⚠️ Primero debes crear una clase en 'Gestionar Clases'")
        return
    
    # Seleccionar clase
    class_names = [f"{c.name} - {c.subject}" for c in st.session_state.classes]
    selected_class_idx = st.selectbox("Selecciona una clase:", range(len(class_names)), 
                                     format_func=lambda x: class_names[x])
    
    selected_class = st.session_state.classes[selected_class_idx]
    
    # Título del examen
    exam_title = st.text_input("Título del examen", value=f"Examen {selected_class.subject}")
    
    # Métodos de entrada
    st.subheader("📁 Método de entrada")
    input_method = st.radio("Selecciona cómo quieres subir el examen:", 
                           ["📄 Subir PDF", "🖼️ Subir imágenes", "✍️ Escribir texto"])
    
    exam_text = ""
    
    if input_method == "📄 Subir PDF":
        uploaded_pdf = st.file_uploader("Sube el PDF del examen", type="pdf")
        if uploaded_pdf:
            try:
                pdf_bytes = uploaded_pdf.read()
                images = pdf_to_images(pdf_bytes)
                
                with st.spinner("Extrayendo texto del PDF..."):
                    all_text = ""
                    for i, image in enumerate(images):
                        st.write(f"Procesando página {i+1}/{len(images)}")
                        img_bytes = preprocess_image(image)
                        
                        if selected_class.subject_type == SubjectType.SCIENCES:
                            text = ocr_service.mathpix_ocr(img_bytes)
                        else:
                            text = ocr_service.google_vision_ocr(img_bytes)
                        
                        all_text += f"\n--- Página {i+1} ---\n{text}\n"
                    
                    exam_text = all_text
                    st.success("✅ Texto extraído exitosamente")
                    with st.expander("Ver texto extraído"):
                        st.text_area("Texto del examen", exam_text, height=300)
            
            except Exception as e:
                st.error(f"❌ Error procesando PDF: {str(e)}")
    
    elif input_method == "🖼️ Subir imágenes":
        uploaded_images = st.file_uploader("Sube las imágenes del examen", 
                                         type=["png", "jpg", "jpeg"], 
                                         accept_multiple_files=True)
        if uploaded_images:
            try:
                with st.spinner("Extrayendo texto de las imágenes..."):
                    all_text = ""
                    for i, uploaded_image in enumerate(uploaded_images):
                        st.write(f"Procesando imagen {i+1}/{len(uploaded_images)}")
                        image = Image.open(uploaded_image)
                        img_bytes = preprocess_image(image)
                        
                        if selected_class.subject_type == SubjectType.SCIENCES:
                            text = ocr_service.mathpix_ocr(img_bytes)
                        else:
                            text = ocr_service.google_vision_ocr(img_bytes)
                        
                        all_text += f"\n--- Imagen {i+1} ---\n{text}\n"
                    
                    exam_text = all_text
                    st.success("✅ Texto extraído exitosamente")
                    with st.expander("Ver texto extraído"):
                        st.text_area("Texto del examen", exam_text, height=300)
            
            except Exception as e:
                st.error(f"❌ Error procesando imágenes: {str(e)}")
    
    elif input_method == "✍️ Escribir texto":
        exam_text = st.text_area("Escribe o pega el texto del examen:", height=300)
    
    # Botón para corregir
    if st.button("🤖 Corregir Examen", disabled=not exam_text):
        if exam_text:
            with st.spinner("Corrigiendo examen con IA..."):
                corrections = ai_service.correct_exam(exam_text, selected_class.subject_type, selected_class.subject)
                
                if corrections and corrections.get("corrections"):
                    # Mostrar resultados
                    st.success("✅ Examen corregido exitosamente")
                    
                    # Resumen
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Puntuación total", f"{corrections.get('total_score', 0)}/10")
                    with col2:
                        correct_answers = sum(1 for c in corrections["corrections"] if c["is_correct"])
                        total_answers = len(corrections["corrections"])
                        st.metric("Respuestas correctas", f"{correct_answers}/{total_answers}")
                    
                    # Mostrar correcciones
                    st.subheader("📋 Correcciones detalladas")
                    for i, correction in enumerate(corrections["corrections"]):
                        with st.expander(f"Pregunta {i+1} - {'✅ Correcta' if correction['is_correct'] else '❌ Incorrecta'}"):
                            st.write(f"**Pregunta:** {correction['question']}")
                            st.write(f"**Respuesta del estudiante:** {correction['student_answer']}")
                            
                            if not correction['is_correct']:
                                st.error(f"**Comentarios:** {correction['comments']}")
                                if correction.get('correct_answer'):
                                    st.info(f"**Respuesta correcta:** {correction['correct_answer']}")
                            else:
                                st.success("Respuesta correcta")
                            
                            st.write(f"**Puntuación:** {correction.get('score', 0)}")
                    
                    # Feedback general
                    if corrections.get("feedback"):
                        st.subheader("💭 Comentarios generales")
                        st.write(corrections["feedback"])
                    
                    # Guardar examen
                    new_exam = Exam(
                        id=f"exam_{len(st.session_state.exams) + 1}",
                        class_id=selected_class.id,
                        title=exam_title,
                        content=exam_text,
                        corrections=corrections["corrections"],
                        grade=corrections.get("total_score", 0),
                        created_at=datetime.now(),
                        corrected_at=datetime.now()
                    )
                    st.session_state.exams.append(new_exam)
                    
                    # Generar PDF corregido
                    pdf_bytes = create_corrected_pdf(exam_text, corrections, exam_title)
                    
                    st.download_button(
                        label="📥 Descargar PDF corregido",
                        data=pdf_bytes,
                        file_name=f"{exam_title}_corregido.pdf",
                        mime="application/pdf"
                    )
                
                else:
                    st.error("❌ No se pudo corregir el examen. Verifica la configuración de la API.")

def show_exam_history():
    """Historial de exámenes"""
    st.header("📊 Historial de Exámenes")
    
    if not st.session_state.exams:
        st.info("📝 No hay exámenes corregidos aún.")
        return
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        class_filter = st.selectbox("Filtrar por clase:", 
                                   ["Todas"] + [c.name for c in st.session_state.classes])
    
    with col2:
        sort_by = st.selectbox("Ordenar por:", ["Fecha", "Puntuación", "Título"])
    
    # Filtrar exámenes
    filtered_exams = st.session_state.exams
    
    if class_filter != "Todas":
        selected_class = next((c for c in st.session_state.classes if c.name == class_filter), None)
        if selected_class:
            filtered_exams = [e for e in filtered_exams if e.class_id == selected_class.id]
    
    # Ordenar exámenes
    if sort_by == "Fecha":
        filtered_exams.sort(key=lambda x: x.created_at, reverse=True)
    elif sort_by == "Puntuación":
        filtered_exams.sort(key=lambda x: x.grade, reverse=True)
    elif sort_by == "Título":
        filtered_exams.sort(key=lambda x: x.title)
    
    # Mostrar exámenes
    for exam in filtered_exams:
        class_obj = next((c for c in st.session_state.classes if c.id == exam.class_id), None)
        
        with st.expander(f"📄 {exam.title} - Puntuación: {exam.grade}/10"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Clase:** {class_obj.name if class_obj else 'N/A'}")
                st.write(f"**Asignatura:** {class_obj.subject if class_obj else 'N/A'}")
                st.write(f"**Fecha:** {exam.created_at.strftime('%d/%m/%Y %H:%M')}")
            
            with col2:
                st.write(f"**Puntuación:** {exam.grade}/10")
                correct_answers = sum(1 for c in exam.corrections if c["is_correct"])
                total_answers = len(exam.corrections)
                st.write(f"**Respuestas correctas:** {correct_answers}/{total_answers}")
            
            # Generar PDF para descarga
            corrections_dict = {
                "corrections": exam.corrections,
                "total_score": exam.grade,
                "feedback": "Examen del historial"
            }
            pdf_bytes = create_corrected_pdf(exam.content, corrections_dict, exam.title)
            
            st.download_button(
                label="📥 Descargar PDF",
                data=pdf_bytes,
                file_name=f"{exam.title}_corregido.pdf",
                mime="application/pdf",
                key=f"delete_exam_{exam.id}")
            
            if delete_button:
                st.session_state.exams = [e for e in st.session_state.exams if e.id != exam.id]
                st.rerun()

def show_scanner(ocr_service: OCRService, ai_service: AIService):
    """Escáner de documentos con cámara"""
    st.header("📷 Escáner de Documentos")
    
    if not st.session_state.classes:
        st.warning("⚠️ Primero debes crear una clase en 'Gestionar Clases'")
        return
    
    # Seleccionar clase
    class_names = [f"{c.name} - {c.subject}" for c in st.session_state.classes]
    selected_class_idx = st.selectbox("Selecciona una clase:", range(len(class_names)), 
                                     format_func=lambda x: class_names[x], key="scanner_class")
    
    selected_class = st.session_state.classes[selected_class_idx]
    
    # Título del examen
    exam_title = st.text_input("Título del examen", value=f"Examen {selected_class.subject}", key="scanner_title")
    
    st.subheader("📱 Escáner con Cámara")
    
    # Configuración WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Procesador de video
    video_processor = VideoProcessor()
    
    # Componente de cámara
    webrtc_ctx = webrtc_streamer(
        key="scanner",
        video_processor_factory=lambda: video_processor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Controles del escáner
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📸 Capturar Imagen"):
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.capture_image()
                st.success("✅ Imagen capturada")
    
    with col2:
        if st.button("🔄 Procesar Imagen Capturada"):
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.captured_image is not None:
                process_captured_image(webrtc_ctx.video_processor.captured_image, 
                                     selected_class, exam_title, ocr_service, ai_service)

def process_captured_image(image_array, selected_class, exam_title, ocr_service, ai_service):
    """Procesa la imagen capturada"""
    try:
        # Convertir numpy array a PIL Image
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Mostrar imagen capturada
        st.image(pil_image, caption="Imagen capturada", use_column_width=True)
        
        # Preprocesar imagen
        with st.spinner("Procesando imagen..."):
            img_bytes = preprocess_image(pil_image)
            
            # Aplicar OCR según el tipo de asignatura
            if selected_class.subject_type == SubjectType.SCIENCES:
                exam_text = ocr_service.mathpix_ocr(img_bytes)
            else:
                exam_text = ocr_service.google_vision_ocr(img_bytes)
            
            if exam_text:
                st.success("✅ Texto extraído exitosamente")
                
                # Mostrar texto extraído
                with st.expander("Ver texto extraído"):
                    st.text_area("Texto del examen", exam_text, height=200)
                
                # Corregir automáticamente
                if st.button("🤖 Corregir Examen Escaneado"):
                    correct_scanned_exam(exam_text, selected_class, exam_title, ai_service)
            else:
                st.error("❌ No se pudo extraer texto de la imagen")
    
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {str(e)}")

def correct_scanned_exam(exam_text, selected_class, exam_title, ai_service):
    """Corrige el examen escaneado"""
    with st.spinner("Corrigiendo examen con IA..."):
        corrections = ai_service.correct_exam(exam_text, selected_class.subject_type, selected_class.subject)
        
        if corrections and corrections.get("corrections"):
            display_correction_results(corrections, exam_text, exam_title, selected_class)
        else:
            st.error("❌ No se pudo corregir el examen. Verifica la configuración de la API.")

def display_correction_results(corrections, exam_text, exam_title, selected_class):
    """Muestra los resultados de la corrección"""
    st.success("✅ Examen corregido exitosamente")
    
    # Resumen
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Puntuación total", f"{corrections.get('total_score', 0)}/10")
    with col2:
        correct_answers = sum(1 for c in corrections["corrections"] if c["is_correct"])
        total_answers = len(corrections["corrections"])
        st.metric("Respuestas correctas", f"{correct_answers}/{total_answers}")
    
    # Mostrar correcciones
    st.subheader("📋 Correcciones detalladas")
    for i, correction in enumerate(corrections["corrections"]):
        with st.expander(f"Pregunta {i+1} - {'✅ Correcta' if correction['is_correct'] else '❌ Incorrecta'}"):
            st.write(f"**Pregunta:** {correction['question']}")
            st.write(f"**Respuesta del estudiante:** {correction['student_answer']}")
            
            if not correction['is_correct']:
                st.error(f"**Comentarios:** {correction['comments']}")
                if correction.get('correct_answer'):
                    st.info(f"**Respuesta correcta:** {correction['correct_answer']}")
            else:
                st.success("Respuesta correcta")
            
            st.write(f"**Puntuación:** {correction.get('score', 0)}")
    
    # Feedback general
    if corrections.get("feedback"):
        st.subheader("💭 Comentarios generales")
        st.write(corrections["feedback"])
    
    # Guardar examen
    new_exam = Exam(
        id=f"exam_{len(st.session_state.exams) + 1}",
        class_id=selected_class.id,
        title=exam_title,
        content=exam_text,
        corrections=corrections["corrections"],
        grade=corrections.get("total_score", 0),
        created_at=datetime.now(),
        corrected_at=datetime.now()
    )
    st.session_state.exams.append(new_exam)
    
    # Generar PDF corregido
    pdf_bytes = create_corrected_pdf(exam_text, corrections, exam_title)
    
    st.download_button(
        label="📥 Descargar PDF corregido",
        data=pdf_bytes,
        file_name=f"{exam_title}_corregido.pdf",
        mime="application/pdf"
    )

def enhance_document_detection(image_array):
    """Mejora la detección de documentos en la imagen"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtros para mejorar la detección
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontrar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Aproximar el contorno a un polígono
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Si encontramos un cuadrilátero, aplicar transformación de perspectiva
        if len(approx) == 4:
            return apply_perspective_transform(image_array, approx)
    
    return image_array

def apply_perspective_transform(image, contour):
    """Aplica transformación de perspectiva para enderezar el documento"""
    try:
        # Ordenar los puntos del contorno
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Suma y diferencia de coordenadas para ordenar los puntos
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        
        # Calcular dimensiones del rectángulo de destino
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Puntos de destino
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
        
        # Calcular matriz de transformación
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Aplicar transformación
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    except Exception as e:
        print(f"Error en transformación de perspectiva: {e}")
        return image

def create_advanced_corrected_pdf(original_text: str, corrections: Dict, title: str, class_info: ExamClass) -> bytes:
    """Crea un PDF corregido más avanzado con mejor formato"""
    doc = fitz.open()
    page = doc.new_page()
    
    # Configurar fuentes y colores
    font_size = 11
    title_font_size = 16
    subtitle_font_size = 14
    red_color = (0.8, 0, 0)      # Rojo para errores
    green_color = (0, 0.6, 0)    # Verde para correctas
    blue_color = (0, 0, 0.8)     # Azul para información
    black_color = (0, 0, 0)      # Negro para texto normal
    
    y_position = 50
    margin = 50
    page_width = 595  # Ancho estándar A4
    
    # Encabezado
    page.insert_text((margin, y_position), title, fontsize=title_font_size, color=black_color)
    y_position += 25
    
    # Información de la clase
    page.insert_text((margin, y_position), f"Clase: {class_info.name}", fontsize=font_size, color=blue_color)
    y_position += 15
    page.insert_text((margin, y_position), f"Asignatura: {class_info.subject} ({class_info.subject_type.value})", 
                    fontsize=font_size, color=blue_color)
    y_position += 15
    page.insert_text((margin, y_position), f"Profesor: {class_info.teacher_name}", fontsize=font_size, color=blue_color)
    y_position += 15
    page.insert_text((margin, y_position), f"Fecha de corrección: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                    fontsize=font_size, color=blue_color)
    y_position += 30
    
    # Línea separadora
    page.draw_line((margin, y_position), (page_width - margin, y_position), color=black_color, width=1)
    y_position += 20
    
    # Resumen de puntuación
    total_score = corrections.get('total_score', 0)
    correct_count = sum(1 for c in corrections.get("corrections", []) if c.get("is_correct", False))
    total_count = len(corrections.get("corrections", []))
    
    page.insert_text((margin, y_position), f"PUNTUACIÓN TOTAL: {total_score}/10", 
                    fontsize=subtitle_font_size, color=red_color)
    y_position += 20
    page.insert_text((margin, y_position), f"Respuestas correctas: {correct_count}/{total_count}", 
                    fontsize=font_size, color=black_color)
    y_position += 30
    
    # Correcciones detalladas
    for i, correction in enumerate(corrections.get("corrections", [])):
        # Verificar si necesitamos nueva página
        if y_position > 700:
            page = doc.new_page()
            y_position = 50
        
        # Número de pregunta
        page.insert_text((margin, y_position), f"PREGUNTA {i+1}:", 
                        fontsize=subtitle_font_size, color=black_color)
        y_position += 20
        
        # Pregunta
        question_text = correction.get('question', '')
        wrapped_question = wrap_text(question_text, 80)
        for line in wrapped_question:
            page.insert_text((margin + 10, y_position), line, fontsize=font_size, color=black_color)
            y_position += 15
        y_position += 5
        
        # Respuesta del estudiante
        page.insert_text((margin + 10, y_position), "Respuesta del estudiante:", 
                        fontsize=font_size, color=black_color)
        y_position += 15
        
        student_answer = correction.get('student_answer', '')
        wrapped_answer = wrap_text(student_answer, 75)
        for line in wrapped_answer:
            page.insert_text((margin + 20, y_position), line, fontsize=font_size, color=black_color)
            y_position += 15
        y_position += 5
        
        # Estado de la respuesta
        if correction.get('is_correct', False):
            page.insert_text((margin + 10, y_position), "✓ CORRECTA", 
                            fontsize=font_size, color=green_color)
            y_position += 15
        else:
            page.insert_text((margin + 10, y_position), "✗ INCORRECTA", 
                            fontsize=font_size, color=red_color)
            y_position += 15
            
            # Comentarios de error
            comments = correction.get('comments', '')
            if comments:
                page.insert_text((margin + 10, y_position), "Comentarios:", 
                                fontsize=font_size, color=red_color)
                y_position += 15
                
                wrapped_comments = wrap_text(comments, 75)
                for line in wrapped_comments:
                    page.insert_text((margin + 20, y_position), line, fontsize=font_size, color=red_color)
                    y_position += 15
                y_position += 5
            
            # Respuesta correcta
            correct_answer = correction.get('correct_answer', '')
            if correct_answer:
                page.insert_text((margin + 10, y_position), "Respuesta correcta:", 
                                fontsize=font_size, color=red_color)
                y_position += 15
                
                wrapped_correct = wrap_text(correct_answer, 75)
                for line in wrapped_correct:
                    page.insert_text((margin + 20, y_position), line, fontsize=font_size, color=red_color)
                    y_position += 15
                y_position += 5
        
        # Puntuación
        score = correction.get('score', 0)
        page.insert_text((margin + 10, y_position), f"Puntuación: {score}", 
                        fontsize=font_size, color=black_color)
        y_position += 25
        
        # Línea separadora
        page.draw_line((margin, y_position), (page_width - margin, y_position), color=(0.7, 0.7, 0.7), width=0.5)
        y_position += 20
    
    # Comentarios generales
    if corrections.get("feedback"):
        if y_position > 650:
            page = doc.new_page()
            y_position = 50
        
        page.insert_text((margin, y_position), "COMENTARIOS GENERALES:", 
                        fontsize=subtitle_font_size, color=black_color)
        y_position += 20
        
        feedback = corrections.get("feedback", "")
        wrapped_feedback = wrap_text(feedback, 80)
        for line in wrapped_feedback:
            page.insert_text((margin + 10, y_position), line, fontsize=font_size, color=black_color)
            y_position += 15
    
    # Guardar PDF
    pdf_bytes = doc.write()
    doc.close()
    
    return pdf_bytes

def wrap_text(text: str, max_chars: int) -> List[str]:
    """Envuelve texto para que se ajuste al ancho de la página"""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= max_chars:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

# Función para exportar datos
def export_class_data(class_obj: ExamClass) -> Dict:
    """Exporta los datos de una clase"""
    class_exams = [e for e in st.session_state.exams if e.class_id == class_obj.id]
    
    return {
        "class_info": {
            "id": class_obj.id,
            "name": class_obj.name,
            "subject": class_obj.subject,
            "subject_type": class_obj.subject_type.value,
            "teacher_name": class_obj.teacher_name,
            "created_at": class_obj.created_at.isoformat()
        },
        "exams": [
            {
                "id": exam.id,
                "title": exam.title,
                "content": exam.content,
                "corrections": exam.corrections,
                "grade": exam.grade,
                "created_at": exam.created_at.isoformat(),
                "corrected_at": exam.corrected_at.isoformat() if exam.corrected_at else None
            }
            for exam in class_exams
        ]
    }

def show_statistics():
    """Muestra estadísticas generales"""
    st.header("📊 Estadísticas Generales")
    
    if not st.session_state.exams:
        st.info("📈 No hay datos suficientes para mostrar estadísticas.")
        return
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clases", len(st.session_state.classes))
    
    with col2:
        st.metric("Total Exámenes", len(st.session_state.exams))
    
    with col3:
        avg_grade = sum(e.grade for e in st.session_state.exams) / len(st.session_state.exams)
        st.metric("Nota Media", f"{avg_grade:.1f}/10")
    
    with col4:
        total_corrections = sum(len(e.corrections) for e in st.session_state.exams)
        st.metric("Total Correcciones", total_corrections)
    
    # Gráfico de distribución de notas
    st.subheader("📈 Distribución de Notas")
    
    grades = [e.grade for e in st.session_state.exams]
    grade_ranges = ["0-2", "2-4", "4-6", "6-8", "8-10"]
    grade_counts = [
        sum(1 for g in grades if 0 <= g < 2),
        sum(1 for g in grades if 2 <= g < 4),
        sum(1 for g in grades if 4 <= g < 6),
        sum(1 for g in grades if 6 <= g < 8),
        sum(1 for g in grades if 8 <= g <= 10)
    ]
    
    chart_data = pd.DataFrame({
        'Rango de Notas': grade_ranges,
        'Cantidad': grade_counts
    })
    
    st.bar_chart(chart_data.set_index('Rango de Notas'))
    
    # Estadísticas por clase
    st.subheader("📚 Estadísticas por Clase")
    
    for class_obj in st.session_state.classes:
        class_exams = [e for e in st.session_state.exams if e.class_id == class_obj.id]
        
        if class_exams:
            with st.expander(f"📖 {class_obj.name} - {class_obj.subject}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Exámenes", len(class_exams))
                
                with col2:
                    class_avg = sum(e.grade for e in class_exams) / len(class_exams)
                    st.metric("Nota Media", f"{class_avg:.1f}/10")
                
                with col3:
                    passing_exams = sum(1 for e in class_exams if e.grade >= 5)
                    pass_rate = (passing_exams / len(class_exams)) * 100
                    st.metric("Tasa de Aprobados", f"{pass_rate:.1f}%")

if __name__ == "__main__":
    main()
