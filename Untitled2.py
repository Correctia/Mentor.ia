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

# API Keys y configuraciones
DEEPSEEK_API_KEY = "sk-2193b6a84e2d428e963633e213d1c439"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Configuración Azure Computer Vision - ACTUALIZA ESTOS VALORES
AZURE_VISION_ENDPOINT = "https://tu-recurso.cognitiveservices.azure.com/"
AZURE_VISION_KEY = "2XR9XUdGP51RIPSoBysjZBECVgZfs9oOUUpnIVdoHyrcDNgsYY3wJQQJ99BGACYeBjFXJ3w3AAAFACOGtAHx"

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

import cv2
import numpy as np
import requests
import time
from PIL import Image
import io

class ImprovedMicrosoftOCR:
    def __init__(self, endpoint, key):
        self.endpoint = endpoint
        self.key = key
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.key,
            'Content-Type': 'application/octet-stream'
        }
        # Usar la versión más reciente para mejor reconocimiento
        self.read_url = f"{self.endpoint}vision/v4.0/read/analyze"
    
    def validate_image_quality(self, image_data):
        """Valida calidad de imagen antes de OCR"""
        try:
            # Convertir a numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return False, "No se pudo cargar la imagen"
            
            height, width = img.shape[:2]
            
            # Verificar resolución mínima
            if width < 800 or height < 600:
                return False, f"Resolución muy baja: {width}x{height}. Mínimo recomendado: 800x600"
            
            # Verificar tamaño de archivo
            if len(image_data) > 20 * 1024 * 1024:  # 20MB
                return False, "Archivo muy grande (>20MB)"
            
            if len(image_data) < 50 * 1024:  # 50KB
                return False, "Archivo muy pequeño (<50KB). Posible imagen muy comprimida"
            
            # Verificar calidad general (detección de borrosidad)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 100:
                return False, f"Imagen muy borrosa (score: {blur_score:.1f}). Mínimo: 100"
            
            return True, f"Imagen válida: {width}x{height}, blur_score: {blur_score:.1f}"
            
        except Exception as e:
            return False, f"Error validando imagen: {str(e)}"
    
    def enhance_image_for_ocr(self, image_data):
        """Mejora imagen específicamente para OCR de escritura manual"""
        try:
            # Convertir a numpy
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return image_data
            
            # 1. Redimensionar si es muy grande (mantener aspect ratio)
            height, width = img.shape[:2]
            if width > 2000 or height > 2000:
                scale = min(2000/width, 2000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 2. Convertir a escala de grises
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # 3. Corrección de perspectiva básica (si detecta bordes)
            gray = self.correct_perspective(gray)
            
            # 4. Mejorar contraste específico para texto
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 5. Reducir ruido preservando texto
            # Filtro bilateral (reduce ruido pero mantiene bordes)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 6. Sharpening específico para texto
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 7. Binarización adaptativa para texto manuscrito
            # Usar dos métodos y combinar
            binary1 = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 10
            )
            
            binary2 = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 15, 10
            )
            
            # Combinar ambos métodos
            combined = cv2.bitwise_and(binary1, binary2)
            
            # 8. Morfología para limpiar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # 9. Convertir de vuelta a bytes
            success, buffer = cv2.imencode('.png', cleaned, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if success:
                return buffer.tobytes()
            else:
                return image_data
                
        except Exception as e:
            print(f"Error en mejora de imagen: {str(e)}")
            return image_data
    
    def correct_perspective(self, gray):
        """Corrección básica de perspectiva"""
        try:
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Buscar el contorno más grande (probablemente la hoja)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Aproximar a rectángulo
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Si encontramos 4 puntos, corregir perspectiva
                if len(approx) == 4:
                    # Ordenar puntos
                    pts = approx.reshape(4, 2)
                    rect = np.zeros((4, 2), dtype="float32")
                    
                    # Top-left tiene la suma más pequeña
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    
                    # Top-right tiene la diferencia más pequeña
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    
                    # Calcular dimensiones del rectángulo corregido
                    width = max(
                        np.linalg.norm(rect[0] - rect[1]),
                        np.linalg.norm(rect[2] - rect[3])
                    )
                    height = max(
                        np.linalg.norm(rect[1] - rect[2]),
                        np.linalg.norm(rect[3] - rect[0])
                    )
                    
                    # Puntos destino
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")
                    
                    # Aplicar transformación de perspectiva
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(gray, M, (int(width), int(height)))
                    
                    return warped
            
            return gray
            
        except Exception as e:
            print(f"Error en corrección de perspectiva: {str(e)}")
            return gray
    
    def extract_text_with_validation(self, image_data):
        """Extracción de texto con validación completa"""
        try:
            # 1. Validar calidad de imagen
            is_valid, message = self.validate_image_quality(image_data)
            if not is_valid:
                return None, f"Imagen no válida: {message}"
            
            print(f"✅ Validación: {message}")
            
            # 2. Mejorar imagen para OCR
            enhanced_image = self.enhance_image_for_ocr(image_data)
            
            # 3. Configurar parámetros específicos para escritura manual
            params = {
                'language': 'es',
                'readingOrder': 'natural',
                'model-version': 'latest',  # Usar modelo más reciente
                'pages': '1'
            }
            
            # 4. Enviar a Microsoft OCR
            response = requests.post(
                self.read_url,
                headers=self.headers,
                data=enhanced_image,
                params=params,
                timeout=30
            )
            
            if response.status_code != 202:
                return None, f"Error OCR: {response.status_code} - {response.text}"
            
            # 5. Obtener resultado
            operation_url = response.headers.get('Operation-Location')
            if not operation_url:
                return None, "No se recibió URL de operación"
            
            # 6. Polling para resultado
            for attempt in range(60):  # Aumentar intentos
                time.sleep(2)
                
                result_response = requests.get(
                    operation_url,
                    headers={'Ocp-Apim-Subscription-Key': self.key},
                    timeout=30
                )
                
                if result_response.status_code == 200:
                    result = result_response.json()
                    status = result.get('status', '')
                    
                    if status == 'succeeded':
                        text, confidence_info = self.extract_text_with_confidence(result)
                        return text, confidence_info
                    elif status == 'failed':
                        return None, "OCR falló al procesar"
                
                print(f"Intento {attempt + 1}/60...")
            
            return None, "Timeout en OCR"
            
        except Exception as e:
            return None, f"Error en OCR: {str(e)}"
    
    def extract_text_with_confidence(self, result):
        """Extrae texto con información de confianza"""
        text_lines = []
        total_confidence = 0
        line_count = 0
        low_confidence_count = 0
        
        analyze_result = result.get('analyzeResult', {})
        read_results = analyze_result.get('readResults', [])
        
        for read_result in read_results:
            lines = read_result.get('lines', [])
            for line in lines:
                line_text = line.get('text', '')
                confidence = line.get('confidence', 0)
                
                line_count += 1
                total_confidence += confidence
                
                if confidence > 0.5:  # Umbral de confianza
                    text_lines.append(line_text)
                elif confidence > 0.3:
                    text_lines.append(f"[?] {line_text}")
                    low_confidence_count += 1
                else:
                    text_lines.append(f"[??] {line_text}")
                    low_confidence_count += 1
        
        extracted_text = '\n'.join(text_lines)
        
        # Información de confianza
        avg_confidence = total_confidence / line_count if line_count > 0 else 0
        quality_ratio = (line_count - low_confidence_count) / line_count if line_count > 0 else 0
        
        confidence_info = {
            'avg_confidence': avg_confidence,
            'quality_ratio': quality_ratio,
            'total_lines': line_count,
            'low_confidence_lines': low_confidence_count,
            'message': f"Confianza promedio: {avg_confidence:.2f}, Calidad: {quality_ratio:.1%}"
        }
        
        return extracted_text, confidence_info

# Función para mostrar guías de captura
def show_capture_guidelines():
    """Muestra guías para mejor captura de imágenes"""
    guidelines = {
        "📸 Captura de Imagen": [
            "Usa la cámara nativa del teléfono (no WhatsApp)",
            "Configura la cámara en máxima resolución",
            "Usa modo 'Documento' si está disponible"
        ],
        "💡 Iluminación": [
            "Luz natural difusa (cerca de ventana)",
            "Evita sombras y reflejos",
            "Usa lámpara LED blanca si es necesario"
        ],
        "📐 Posicionamiento": [
            "Coloca el examen en superficie plana",
            "Foto perpendicular al papel (90°)",
            "El texto debe ocupar al menos 60% de la imagen"
        ],
        "🔍 Calidad": [
            "Texto nítido y enfocado",
            "Contraste alto (papel blanco, tinta oscura)",
            "Evita fotos movidas o borrosas"
        ],
        "📱 Transferencia": [
            "Evita WhatsApp (comprime imágenes)",
            "Usa cable USB, email o Google Drive",
            "Si usas WhatsApp, envía como 'Documento'"
        ]
    }
    
    return guidelines
    
    def is_configured(self):
        """Verifica si Microsoft OCR está configurado"""
        return (self.endpoint and 
                self.key and 
                self.endpoint != "https://tu-recurso.cognitiveservices.azure.com/" and
                len(self.key) > 10)

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
                text_quality REAL DEFAULT 0,
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
        """Corrector con DeepSeek API y Microsoft OCR mejorado"""
        self.client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.db = DatabaseManager()
        self.microsoft_ocr = MicrosoftOCR()
    
    def extract_text_from_file(self, uploaded_file):
        """Extrae texto de archivos con validación mejorada"""
        try:
            file_type = uploaded_file.type
            ocr_method = "unknown"
            text_quality = 0.0
            
            if file_type == "application/pdf":
                # Extraer texto de PDF
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    
                    # Intentar extraer texto nativo
                    page_text = page.get_text()
                    
                    if page_text.strip() and len(page_text.strip()) > 50:
                        # Texto nativo suficiente
                        text += page_text
                        ocr_method = "pdf_native"
                        text_quality = 0.9
                    else:
                        # Usar OCR en la página
                        if self.microsoft_ocr.is_configured():
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Mayor resolución
                            img_data = pix.tobytes("png")
                            
                            ocr_text = self.microsoft_ocr.extract_text_from_image(img_data)
                            if ocr_text and len(ocr_text.strip()) > 10:
                                text += ocr_text + "\n"
                                ocr_method = "microsoft_ocr"
                                text_quality = 0.7
                            else:
                                st.warning(f"OCR no pudo extraer texto de la página {page_num + 1}")
                        else:
                            st.warning("Microsoft OCR no configurado")
                            ocr_method = "not_configured"
                
                pdf_document.close()
                
                # Validar calidad del texto extraído
                if len(text.strip()) < 20:
                    st.error("Texto extraído muy corto. Verifica la calidad de la imagen.")
                    return None, "insufficient_text", 0.0
                
                return text, ocr_method, text_quality
            
            elif file_type.startswith("image/"):
                # OCR para imágenes
                if self.microsoft_ocr.is_configured():
                    image_data = uploaded_file.read()
                    
                    # Verificar tamaño de imagen
                    if len(image_data) > 20 * 1024 * 1024:  # 20MB
                        st.error("Imagen muy grande. Máximo 20MB.")
                        return None, "file_too_large", 0.0
                    
                    text = self.microsoft_ocr.extract_text_from_image(image_data)
                    ocr_method = "microsoft_ocr"
                    
                    if not text:
                        st.error("No se pudo extraer texto. Verifica que la imagen sea legible.")
                        return None, "ocr_failed", 0.0
                    
                    # Evaluar calidad del texto
                    if len(text.strip()) < 20:
                        st.warning("Texto extraído muy corto. La escritura puede ser ilegible.")
                        text_quality = 0.3
                    elif "[?]" in text:
                        st.warning("Algunas partes del texto tienen baja confianza.")
                        text_quality = 0.5
                    else:
                        text_quality = 0.8
                    
                    return text, ocr_method, text_quality
                else:
                    st.error("Microsoft OCR no configurado. No se puede procesar imágenes.")
                    return None, "not_configured", 0.0
            
            else:
                # Archivo de texto
                text = str(uploaded_file.read(), "utf-8")
                return text, "text_file", 1.0
                
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            return None, "error", 0.0
    
    def validate_extracted_text(self, text):
        """Valida la calidad del texto extraído"""
        if not text or len(text.strip()) < 10:
            return False, "Texto demasiado corto"
        
        # Verificar si hay contenido coherente
        words = text.split()
        if len(words) < 5:
            return False, "Muy pocas palabras extraídas"
        
        # Verificar caracteres especiales excesivos
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,;:!?-')
        if special_chars / len(text) > 0.3:
            return False, "Demasiados caracteres especiales - posible error OCR"
        
        return True, "Texto válido"
    
    def correct_exam(self, exam_text, criteria, rubric, subject="General"):
        """Corrección con IA mejorada"""
        try:
            # Validar texto extraído
            is_valid, validation_msg = self.validate_extracted_text(exam_text)
            if not is_valid:
                st.error(f"Problema con el texto extraído: {validation_msg}")
                return self.create_error_correction(validation_msg)
            
            # Limpiar texto
            cleaned_text = self.clean_extracted_text(exam_text)
            
            # Truncar texto si es muy largo
            max_chars = 6000
            if len(cleaned_text) > max_chars:
                cleaned_text = cleaned_text[:max_chars] + "\n[...texto truncado...]"
            
            system_prompt = f"""Eres un profesor experto en {subject}. 
            
IMPORTANTE: Debes evaluar el examen basándote en el contenido extraído, aunque tenga errores de OCR.
Ignora errores tipográficos obvios causados por OCR y enfócate en el contenido académico.

CRITERIOS DE EVALUACIÓN: {criteria}
RÚBRICA: {rubric}

INSTRUCCIONES:
1. Asigna una calificación entre 0 y 100 puntos
2. Si el texto parece ilegible o sin contenido académico, explica por qué
3. Considera el esfuerzo del estudiante aunque haya errores de OCR
4. Proporciona comentarios constructivos

Responde en formato JSON válido."""

            user_prompt = f"""TEXTO DEL EXAMEN:
{cleaned_text}

Evalúa este examen y responde con este formato JSON exacto:
{{
    "nota_final": {{
        "puntuacion": 75,
        "puntuacion_maxima": 100,
        "porcentaje": 75,
        "letra": "B"
    }},
    "evaluaciones": [
        {{
            "seccion": "Análisis del Contenido",
            "puntos": 75,
            "max_puntos": 100,
            "comentario": "Comentario sobre el contenido académico",
            "fortalezas": ["Fortaleza 1", "Fortaleza 2"],
            "mejoras": ["Mejora 1", "Mejora 2"]
        }}
    ],
    "comentario": "Comentario general sobre el examen y la calidad del texto extraído",
    "recomendaciones": ["Recomendación 1", "Recomendación 2"],
    "calidad_texto": "Evaluación de la calidad del texto extraído"
}}"""

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            
            # Limpiar respuesta JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            # Parsear JSON
            try:
                result = json.loads(response_text)
                
                # Validar que tenga la estructura mínima
                if "nota_final" not in result:
                    raise ValueError("Respuesta sin nota_final")
                
                return result
                
            except json.JSONDecodeError as e:
                st.error(f"Error parseando respuesta JSON: {str(e)}")
                return self.create_fallback_correction(exam_text)
            
        except Exception as e:
            st.error(f"Error en corrección: {str(e)}")
            return self.create_fallback_correction(exam_text)
    
    def clean_extracted_text(self, text):
        """Limpia el texto extraído por OCR"""
        # Eliminar caracteres extraños comunes en OCR
        text = text.replace('|', 'l')
        text = text.replace('0', 'o')  # En algunos contextos
        text = text.replace('5', 's')  # En algunos contextos
        
        # Eliminar líneas muy cortas que pueden ser ruido
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 2 and not line.startswith('[?]'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_error_correction(self, error_msg):
        """Crea una corrección de error cuando OCR falla"""
        return {
            "nota_final": {
                "puntuacion": 0,
                "puntuacion_maxima": 100,
                "porcentaje": 0,
                "letra": "F"
            },
            "evaluaciones": [{
                "seccion": "Error de Procesamiento",
                "puntos": 0,
                "max_puntos": 100,
                "comentario": f"No se pudo evaluar el examen: {error_msg}",
                "fortalezas": [],
                "mejoras": ["Verificar calidad de la imagen", "Usar imagen más clara"]
            }],
            "comentario": f"Error en procesamiento: {error_msg}",
            "recomendaciones": [
                "Verificar que la imagen sea clara y legible",
                "Asegurar buena iluminación",
                "Usar mayor resolución",
                "Verificar que el texto sea lo suficientemente grande"
            ],
            "calidad_texto": "Error en extracción"
        }
    
    def create_fallback_correction(self, exam_text=""):
        """Corrección de emergencia mejorada"""
        # Intentar evaluar longitud del texto
        if len(exam_text.strip()) < 50:
            puntuacion = 30
            letra = "F"
            comentario = "Texto extraído muy corto - posible problema de OCR"
        else:
            puntuacion = 60
            letra = "D"
            comentario = "Evaluación básica - problema en procesamiento avanzado"
        
        return {
            "nota_final": {
                "puntuacion": puntuacion,
                "puntuacion_maxima": 100,
                "porcentaje": puntuacion,
                "letra": letra
            },
            "evaluaciones": [{
                "seccion": "Evaluación Básica",
                "puntos": puntuacion,
                "max_puntos": 100,
                "comentario": comentario,
                "fortalezas": ["Envío completado"],
                "mejoras": ["Mejorar legibilidad", "Verificar calidad de imagen"]
            }],
            "comentario": "Corrección automática básica aplicada",
            "recomendaciones": [
                "Mejorar calidad de la imagen",
                "Verificar configuración OCR",
                "Usar letra más clara"
            ]
        }

    def generate_criteria_from_text(self, text, subject):
        """Genera criterios automáticamente desde texto usando DeepSeek"""
        try:
            # Verificar que el texto sea válido
            if not text or len(text.strip()) < 20:
                return {
                    "criteria": f"Criterios básicos para {subject}",
                    "rubric": "Excelente (90-100), Bueno (70-89), Regular (50-69), Deficiente (0-49)"
                }
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"Eres un experto en {subject}. Genera criterios de evaluación basándote en el texto del examen."},
                    {"role": "user", "content": f"Basándote en este texto de examen de {subject}, genera criterios de evaluación específicos:\n\n{text[:1000]}"}
                ],
                temperature=0.1,
                max_tokens=400
            )
        
            response_text = response.choices[0].message.content
            
            # Extraer criterios y rúbrica
            if "criterios" in response_text.lower():
                parts = response_text.lower().split("criterios")
                if len(parts) > 1:
                    criteria = parts[1].split("rúbrica")[0].strip() if "rúbrica" in parts[1] else parts[1].strip()
                else:
                    criteria = response_text
            else:
                criteria = response_text
            
            # Generar rúbrica estándar
            rubric = f"Rúbrica para {subject}: Excelente (90-100): Dominio completo, Bueno (70-89): Comprensión adecuada, Regular (50-69): Comprensión básica, Deficiente (0-49): No demuestra comprensión"
            
            return {
                "criteria": criteria[:500],  # Limitar longitud
                "rubric": rubric
            }
            
        except Exception as e:
            st.error(f"Error generando criterios: {str(e)}")
            return {
                "criteria": f"Criterios personalizados para {subject}",
                "rubric": f"Rúbrica personalizada para {subject}"
            }
    
    def save_exam_result(self, user_id, group_id, filename, subject, result, ocr_method="unknown", text_quality=0.0):
        """Guarda resultado en base de datos con calidad de texto"""
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
            result['nota_final']['puntuacion'],
            result['nota_final']['puntuacion_maxima'],
            json.dumps(result, ensure_ascii=False),
            ocr_method,
            text_quality
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
        
        # Mostrar información de configuración
        st.info(f"Endpoint: {corrector.microsoft_ocr.endpoint}")
        st.info("API Key: " + "*" * 20 + corrector.microsoft_ocr.key[-4:])
        
        # Test de OCR
        st.subheader("🧪 Probar OCR")
        test_image = st.file_uploader("Sube una imagen para probar OCR", type=['png', 'jpg', 'jpeg'])
        
        if test_image and st.button("Probar OCR"):
            with st.spinner("Procesando imagen..."):
                image_data = test_image.read()
                text = corrector.microsoft_ocr.extract_text_from_image(image_data)
                
                if text:
                    st.success("✅ OCR funcionando correctamente")
                    st.text_area("Texto extraído:", text, height=200)
                else:
                    st.error("❌ Error en OCR")
    else:
        st.error("❌ Microsoft OCR no configurado")
        st.markdown("""
        **Para configurar Microsoft OCR:**
        1. Crear recurso Computer Vision en Azure
        2. Obtener endpoint y API key
        3. Actualizar variables en el código:
           - `AZURE_VISION_ENDPOINT`
           - `AZURE_VISION_KEY`
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
    
    ### Microsoft OCR (Recomendado)
    - Mayor precisión en escritura manual
    - Mejor reconocimiento de caracteres
    - Procesamiento avanzado
    
    ### Configuración:
    1. Crear recurso Computer Vision en Azure
    2. Obtener endpoint y API key
    3. Actualizar variables en el código
    
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
        # Seleccionar asignatura
        subject = st.selectbox(
            "Asignatura",
            list(SUBJECT_COLORS.keys()),
            help="Selecciona la asignatura del examen"
        )
        
        # Seleccionar grupo (si disponible)
        group_id = None
        if user_plan.can_create_groups:
            df_groups = corrector.get_user_groups(user[0])
            if not df_groups.empty:
                group_options = ["Sin grupo"] + df_groups['name'].tolist()
                group_selection = st.selectbox("Grupo", group_options)
                
                if group_selection != "Sin grupo":
                    group_id = df_groups[df_groups['name'] == group_selection]['id'].iloc[0]
    
    with col2:
        # Modo de evaluación
        evaluation_mode = st.radio(
            "Modo de Evaluación",
            ["Automático", "Personalizado"],
            help="Automático: criterios generados por IA | Personalizado: define tus propios criterios"
        )
    
    # Criterios personalizados
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
    
    # Subir archivo
    st.subheader("📤 Subir Examen")
    
    uploaded_file = st.file_uploader(
        "Selecciona el archivo del examen",
        type=['png', 'jpg', 'jpeg', 'pdf', 'txt'],
        help="Formatos soportados: PNG, JPG, JPEG, PDF, TXT"
    )
    
    if uploaded_file is not None:
        st.success(f"Archivo cargado: {uploaded_file.name}")
        
        # Mostrar preview si es imagen
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Vista previa", use_column_width=True)
        
        # Botón para procesar
        if st.button("🚀 Procesar Examen", type="primary"):
            with st.spinner("Procesando examen..."):
                
                # Extraer texto
                st.info("Extrayendo texto del archivo...")
                text, ocr_method, text_quality = corrector.extract_text_from_file(uploaded_file)
                
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

