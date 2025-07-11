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
    GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", "AIzaSyAyGT7uDH5Feaqtc27fcF7ArgkrRO8jU0Q")
except:
    GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "AIzaSyAyGT7uDH5Feaqtc27fcF7ArgkrRO8jU0Q")

# Configuración de DeepSeek API con manejo de errores
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
import base64
import json

class ImprovedGoogleOCR:
    def __init__(self, api_key):
        """Inicialización con validación de API key"""
        self.api_key = api_key
        self.is_available = False
        self.error_message = None
        
        # Validar API key
        if not api_key:
            self.error_message = "Google Vision API key no configurada"
            st.warning("⚠️ Google Vision API no configurada. Funcionalidad OCR limitada.")
            return
        
        if len(api_key) < 30:
            self.error_message = "Google Vision API key inválida"
            st.warning("⚠️ Google Vision API key parece inválida.")
            return
        
        self.vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        self.is_available = True
        
    def is_configured(self):
        """Verifica si Google OCR está configurado correctamente"""
        return self.is_available and self.api_key and len(self.api_key) > 30
        
    def validate_image_quality(self, image_data):
        """Valida calidad de imagen antes de OCR - VERSION MEJORADA"""
        try:
        # Convertir a numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
            if img is None:
                return False, "No se pudo cargar la imagen"
        
            height, width = img.shape[:2]
        
        # Verificar resolución mínima (MÁS FLEXIBLE)
            if width < 400 or height < 300:
                return False, f"Resolución muy baja: {width}x{height}. Mínimo recomendado: 400x300"
        
        # Verificar tamaño de archivo (Google Vision API límite: 20MB)
            if len(image_data) > 20 * 1024 * 1024:  # 20MB
                return False, "Archivo muy grande (>20MB)"
        
        # Reducir límite mínimo (MÁS FLEXIBLE)
            if len(image_data) < 10 * 1024:  # 10KB
                return False, "Archivo muy pequeño (<10KB). Posible imagen corrupta"
        
        # Verificar calidad general (MÁS FLEXIBLE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Reducir threshold de blur (MÁS FLEXIBLE)
            if blur_score < 30:
                print(f"⚠️ Imagen posiblemente borrosa (score: {blur_score:.1f}), pero procesando...")
        
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
    
    def extract_text_from_image_debug(self, image_data):
        """Versión con diagnóstico completo para identificar problemas"""
        print("🔍 Iniciando diagnóstico completo de OCR...")
    
    # 1. Verificar configuración
        if not self.is_configured():
            print("❌ Google Vision API no configurada")
            return None, "Google Vision API no configurada"
    
        try:
        # 2. Verificar datos de imagen
            if not image_data:
                print("❌ No se recibieron datos de imagen")
                return None, "No se recibieron datos de imagen"
        
            print(f"📊 Datos recibidos: {len(image_data)} bytes")
        
        # 3. Verificar si es una imagen válida
            try:
                from PIL import Image
                import io
            
            # Intentar abrir la imagen
                image = Image.open(io.BytesIO(image_data))
                print(f"✅ Imagen válida: {image.format}, {image.size}, {image.mode}")
            
            # Verificar tamaño mínimo
                if image.size[0] < 50 or image.size[1] < 50:
                    print(f"❌ Imagen demasiado pequeña: {image.size}")
                    return None, f"Imagen demasiado pequeña: {image.size}"
            
            # Verificar tamaño máximo (Google Vision tiene límites)
                max_size = 4000
                if image.size[0] > max_size or image.size[1] > max_size:
                    print(f"⚠️ Imagen grande: {image.size}, redimensionando...")
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convertir de vuelta a bytes
                    output = io.BytesIO()
                    image.save(output, format='PNG')
                    image_data = output.getvalue()
                    print(f"✅ Imagen redimensionada a: {image.size}")
            
            except Exception as e:
                print(f"❌ Error al procesar imagen: {str(e)}")
                return None, f"Error al procesar imagen: {str(e)}"
        
        # 4. Validar calidad de imagen (si existe el método)
            if hasattr(self, 'validate_image_quality'):
                is_valid, message = self.validate_image_quality(image_data)
                print(f"📊 Validación calidad: {message}")
            
                if not is_valid and "borrosa" not in message.lower():
                    print(f"❌ Imagen rechazada: {message}")
                    return None, f"Imagen no válida: {message}"
        
        # 5. Mejorar imagen para OCR (si existe el método)
            enhanced_image = image_data
            if hasattr(self, 'enhance_image_for_ocr'):
                try:
                    enhanced_image = self.enhance_image_for_ocr(image_data)
                    print("✅ Imagen mejorada para OCR")
                except Exception as e:
                    print(f"⚠️ Error mejorando imagen, usando original: {str(e)}")
                    enhanced_image = image_data
        
        # 6. Codificar imagen en base64
            import base64
            image_base64 = base64.b64encode(enhanced_image).decode('utf-8')
            print(f"✅ Imagen codificada (tamaño: {len(image_base64)} chars)")
        
        # 7. Verificar URL de la API
            if not hasattr(self, 'vision_url') or not self.vision_url:
                print("❌ URL de Google Vision API no configurada")
                return None, "URL de Google Vision API no configurada"
        
            print(f"📡 URL API: {self.vision_url}")
        
        # 8. Configurar request para Google Vision API
            request_payload = {
                "requests": [
                    {
                        "image": {
                            "content": image_base64
                        },
                        "features": [
                            {
                                "type": "DOCUMENT_TEXT_DETECTION",
                                "maxResults": 1
                            }
                        ],
                        "imageContext": {
                            "languageHints": ["es", "en"]
                        }
                    }
                ]
            }
        
            print("📡 Enviando request a Google Vision API...")
        
        # 9. Enviar request con mejor manejo de errores
            import requests
            headers = {
                'Content-Type': 'application/json'
            }
        
            try:
                response = requests.post(
                    self.vision_url,
                    headers=headers,
                    json=request_payload,
                    timeout=60
                )
            
                print(f"📥 Respuesta recibida: {response.status_code}")
            
                if response.status_code != 200:
                    print(f"❌ Error HTTP: {response.status_code}")
                    print(f"❌ Respuesta: {response.text}")
                    return None, f"Error Google Vision API: {response.status_code} - {response.text}"
            
            except requests.exceptions.Timeout:
                print("❌ Timeout en la conexión")
                return None, "Timeout en la conexión a Google Vision API"
            except requests.exceptions.ConnectionError:
                print("❌ Error de conexión")
                return None, "Error de conexión a Google Vision API"
            except Exception as e:
                print(f"❌ Error en request: {str(e)}")
                return None, f"Error en request: {str(e)}"
        
        # 10. Procesar respuesta con mejor validación
            try:
                result = response.json()
                print(f"✅ JSON parseado correctamente")
            
            except Exception as e:
                print(f"❌ Error parseando JSON: {str(e)}")
                print(f"❌ Respuesta raw: {response.text[:500]}...")
                return None, f"Error parseando respuesta JSON: {str(e)}"
        
        # 11. Validar estructura de respuesta
            if 'responses' not in result:
                print("❌ Respuesta no tiene campo 'responses'")
                print(f"❌ Estructura: {list(result.keys())}")
                return None, "Respuesta de Google Vision API inválida"
        
            if len(result['responses']) == 0:
                print("❌ Lista de respuestas vacía")
                return None, "No se recibió respuesta de Google Vision API"
        
            vision_response = result['responses'][0]
        
        # 12. Verificar errores en la respuesta
            if 'error' in vision_response:
                error_msg = vision_response['error'].get('message', 'Error desconocido')
                error_code = vision_response['error'].get('code', 'Sin código')
                print(f"❌ Error en Google Vision: {error_code} - {error_msg}")
                return None, f"Error en Google Vision: {error_code} - {error_msg}"
        
        # 13. Verificar detección de texto
            if 'textAnnotations' not in vision_response:
                print("❌ No hay campo 'textAnnotations' en la respuesta")
                print(f"❌ Campos disponibles: {list(vision_response.keys())}")
                return None, "Google Vision no devolvió anotaciones de texto"
        
            if not vision_response['textAnnotations']:
                print("❌ Lista de textAnnotations vacía")
            # Intentar con fullTextAnnotation
                if 'fullTextAnnotation' in vision_response:
                    full_text = vision_response['fullTextAnnotation'].get('text', '')
                    if full_text.strip():
                        print(f"✅ Texto encontrado en fullTextAnnotation: {len(full_text)} caracteres")
                        return full_text, {
                            'avg_confidence': 0.8,
                            'quality_ratio': 0.8,
                            'total_lines': len(full_text.split('\n')),
                            'low_confidence_lines': 0,
                            'message': "Texto extraído de fullTextAnnotation"
                        }
            
                return None, "No se detectó texto en la imagen. Verifica que:\n- El texto sea legible\n- Haya suficiente contraste\n- La imagen no esté muy borrosa"
        
            print(f"✅ Texto detectado: {len(vision_response['textAnnotations'])} elementos")
        
        # 14. Extraer texto
            text_annotations = vision_response['textAnnotations']
            full_text = text_annotations[0].get('description', '')
        
            if not full_text.strip():
                print("❌ Texto extraído está vacío")
                return None, "El texto extraído está vacío"
        
            print(f"✅ Texto extraído: {len(full_text)} caracteres")
            print(f"📄 Primeros 200 caracteres: {full_text[:200]}...")
        
        # 15. Información de confianza
            confidence_info = {
                'avg_confidence': 0.8,
                'quality_ratio': 0.8,
                'total_lines': len(full_text.split('\n')),
                'low_confidence_lines': 0,
                'message': f"Texto extraído exitosamente: {len(full_text)} caracteres"
            }
        
            return full_text, confidence_info
        
        except Exception as e:
            print(f"❌ Error general: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Error en OCR: {str(e)}"


    def quick_test_ocr(self, image_path):
        """Función de prueba rápida para el OCR"""
        print(f"🧪 Probando OCR con: {image_path}")
    
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
        
            result, info = self.extract_text_from_image_debug(image_data)
        
            if result:
                print(f"✅ OCR exitoso!")
                print(f"📄 Texto: {result[:200]}...")
                print(f"📊 Info: {info}")
            else:
                print(f"❌ OCR falló: {info}")
            
        except Exception as e:
            print(f"❌ Error en test: {str(e)}")


# Función para verificar configuración
    def check_configuration(self):
        """Verifica que todo esté configurado correctamente"""
        print("🔍 Verificando configuración...")
    
    # Verificar API Key
        if not hasattr(self, 'api_key') or not self.api_key:
            print("❌ API Key no configurada")
            return False
    
    # Verificar URL
        if not hasattr(self, 'vision_url') or not self.vision_url:
            print("❌ URL de Vision API no configurada")
            return False
    
    # Verificar que la URL contenga la API key
        if 'key=' not in self.vision_url:
            print("❌ URL no contiene API key")
            return False
    
        print("✅ Configuración OK")
        return True
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
        """Corrector con DeepSeek API y Google OCR mejorado - con manejo de errores"""
        self.client = None
        self.db = None
        self.microsoft_ocr = None
        self.initialization_errors = []
        
        # Inicializar base de datos
        try:
            self.db = DatabaseManager()
        except Exception as e:
            self.initialization_errors.append(f"Error base de datos: {str(e)}")
            st.error(f"Error al inicializar base de datos: {str(e)}")
        
        # Inicializar DeepSeek API
        try:
            if not DEEPSEEK_API_KEY:
                self.initialization_errors.append("DeepSeek API key no configurada")
                st.error("❌ DeepSeek API key no configurada. Por favor configura DEEPSEEK_API_KEY en los secrets o variables de entorno.")
            else:
                self.client = openai.OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL
                )
                # Verificar conexión
                self.test_deepseek_connection()
        except Exception as e:
            self.initialization_errors.append(f"Error DeepSeek API: {str(e)}")
            st.error(f"Error al inicializar DeepSeek API: {str(e)}")
        
        # Inicializar Google OCR
        try:
            if GOOGLE_VISION_API_KEY:
                self.microsoft_ocr = ImprovedGoogleOCR(GOOGLE_VISION_API_KEY)
                if not self.microsoft_ocr.is_configured():
                    self.initialization_errors.append("Google Vision API no configurada correctamente")
            else:
                self.initialization_errors.append("Google Vision API key no encontrada")
                st.warning("⚠️ Google Vision API no configurada. Funcionalidad OCR limitada.")
        except Exception as e:
            self.initialization_errors.append(f"Error Google OCR: {str(e)}")
            st.error(f"Error al inicializar Google OCR: {str(e)}")
    
    def test_deepseek_connection(self):
        """Verificar conexión con DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.initialization_errors.append(f"Error conectando con DeepSeek: {str(e)}")
            st.error(f"Error conectando con DeepSeek API: {str(e)}")
            return False
    
    def is_ready(self):
        """Verifica si el corrector está listo para funcionar"""
        return (self.client is not None and 
                self.db is not None and 
                len(self.initialization_errors) == 0)
    
    def get_initialization_status(self):
        """Retorna el estado de inicialización"""
        status = {
            'deepseek_api': self.client is not None,
            'database': self.db is not None,
            'google_ocr': self.microsoft_ocr is not None and self.microsoft_ocr.is_configured(),
            'errors': self.initialization_errors
        }
        return status
    
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
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Mayor resolución
                        img_data = pix.tobytes("png")
                        
                        ocr_text = self.microsoft_ocr.extract_text_from_image(img_data)
                        if ocr_text and len(ocr_text.strip()) > 10:
                            text += ocr_text + "\n"
                            ocr_method = "google_ocr"
                            text_quality = 0.7
                        else:
                            text += "[Página sin texto reconocido]\n"
                            text_quality = 0.3
                
                pdf_document.close()
                
            elif file_type.startswith("image/"):
                # Procesar imagen
                image_bytes = uploaded_file.read()
                
                # Usar OCR mejorado con validación
                if self.microsoft_ocr and self.microsoft_ocr.is_configured():
                    extracted_text, confidence_info = self.microsoft_ocr.extract_text_with_validation(image_bytes)
                    
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
        """Corrige examen usando DeepSeek API con rúbrica personalizada"""
        if not self.client:
            return None, "DeepSeek API no configurada"
        
        try:
            # Prompt mejorado para corrección
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
                "preguntas_analizadas": [
                    {{
                        "numero": int,
                        "pregunta": "texto de la pregunta",
                        "respuesta_estudiante": "respuesta del estudiante",
                        "puntos_obtenidos": float,
                        "puntos_maximos": float,
                        "es_correcta": boolean,
                        "explicacion": "explicación detallada",
                        "sugerencias": "sugerencias de mejora"
                    }}
                ],
                "resumen_general": {{
                    "fortalezas": ["lista de fortalezas"],
                    "areas_mejora": ["áreas a mejorar"],
                    "recomendaciones": ["recomendaciones específicas"]
                }},
                "tiempo_estimado_estudio": "tiempo recomendado para repasar",
                "recursos_recomendados": ["recursos adicionales"]
            }}
            
            INSTRUCCIONES:
            1. Analiza cada pregunta individualmente
            2. Asigna puntuación parcial cuando sea apropiado
            3. Explica claramente por qué cada respuesta es correcta o incorrecta
            4. Proporciona sugerencias constructivas
            5. Mantén un tono profesional y alentador
            6. Si no puedes identificar preguntas claramente, analiza el contenido general
            """
            
            user_prompt = f"""
            EXAMEN A CORREGIR:
            {text}
            
            Por favor, corrige este examen siguiendo los criterios especificados y devuelve la evaluación en formato JSON.
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            # Procesar respuesta
            response_text = response.choices[0].message.content
            
            # Intentar parsear JSON
            try:
                correction_data = json.loads(response_text)
                return correction_data, None
            except json.JSONDecodeError:
                # Si no es JSON válido, crear estructura básica
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
        """Guarda resultado del examen en la base de datos"""
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

def main():
    """Función principal de la aplicación"""
    st.title("🎓 Mentor.ia - Corrector Inteligente")
    st.markdown("### Corrección automática de exámenes con IA")
    
    # Inicializar corrector
    if 'corrector' not in st.session_state:
        with st.spinner("Inicializando sistema..."):
            st.session_state.corrector = ExamCorrector()
    
    corrector = st.session_state.corrector
    
    # Mostrar estado de inicialización
    status = corrector.get_initialization_status()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if status['deepseek_api']:
            st.success("✅ DeepSeek API")
        else:
            st.error("❌ DeepSeek API")
    
    with col2:
        if status['google_ocr']:
            st.success("✅ Google OCR")
        else:
            st.warning("⚠️ Google OCR")
    
    with col3:
        if status['database']:
            st.success("✅ Base de datos")
        else:
            st.error("❌ Base de datos")
    
    # Mostrar errores si los hay
    if status['errors']:
        with st.expander("⚠️ Errores de inicialización"):
            for error in status['errors']:
                st.error(error)
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selección de materia
        subject = st.selectbox(
            "Materia",
            list(SUBJECT_COLORS.keys()),
            index=0
        )
        
        # Puntuación total
        total_points = st.number_input(
            "Puntuación total",
            min_value=1,
            max_value=100,
            value=10
        )
        
        # Rúbrica personalizada
        custom_rubric = st.text_area(
            "Rúbrica personalizada (opcional)",
            placeholder="Describe los criterios específicos de evaluación..."
        )
        
        # Guías de captura
        with st.expander("📸 Guías de captura"):
            guidelines = show_capture_guidelines()
            for category, tips in guidelines.items():
                st.write(f"**{category}**")
                for tip in tips:
                    st.write(f"• {tip}")
    
    # Interfaz principal
    if not corrector.is_ready():
        st.error("❌ Sistema no está listo. Revisa la configuración de APIs.")
        return
    
    # Carga de archivos
    st.header("📁 Cargar examen")
    
    uploaded_file = st.file_uploader(
        "Sube imagen o PDF del examen",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="Formatos soportados: JPG, PNG, PDF"
    )
    
    if uploaded_file is not None:
        # Mostrar archivo cargado
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        
        # Vista previa si es imagen
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Vista previa", use_column_width=True)
        
        # Procesar archivo
        if st.button("🔍 Procesar y corregir examen", type="primary"):
            with st.spinner("Procesando examen..."):
                
                # Extraer texto
                st.info("📝 Extrayendo texto...")
                text, ocr_method, text_quality = corrector.extract_text_from_file(uploaded_file)
                
                if text:
                    # Mostrar texto extraído
                    with st.expander("📄 Texto extraído"):
                        st.text_area("Texto del examen", text, height=200)
                        st.info(f"Método: {ocr_method} | Calidad: {text_quality:.1%}")
                    
                    # Corregir examen
                    st.info("🤖 Corrigiendo con IA...")
                    correction_data, error = corrector.correct_exam(
                        text, subject, custom_rubric, total_points
                    )
                    
                    if correction_data:
                        # Mostrar resultados
                        st.success("✅ Examen corregido exitosamente")
                        
                        # Métricas principales
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Puntuación",
                                f"{correction_data.get('puntuacion_total', 0)}/{correction_data.get('puntuacion_maxima', total_points)}"
                            )
                        
                        with col2:
                            st.metric(
                                "Porcentaje",
                                f"{correction_data.get('porcentaje', 0):.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Calificación",
                                correction_data.get('calificacion_letra', 'F')
                            )
                        
                        with col4:
                            st.metric(
                                "Calidad OCR",
                                f"{text_quality:.1%}"
                            )
                        
                        # Análisis detallado
                        st.header("📊 Análisis detallado")
                        
                        # Preguntas analizadas
                        if 'preguntas_analizadas' in correction_data:
                            st.subheader("📝 Preguntas analizadas")
                            for i, pregunta in enumerate(correction_data['preguntas_analizadas']):
                                with st.expander(f"Pregunta {pregunta.get('numero', i+1)}"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Pregunta:**")
                                        st.write(pregunta.get('pregunta', 'No identificada'))
                                        
                                        st.write("**Respuesta del estudiante:**")
                                        st.write(pregunta.get('respuesta_estudiante', 'No identificada'))
                                    
                                    with col2:
                                        st.write("**Puntuación:**")
                                        st.write(f"{pregunta.get('puntos_obtenidos', 0)}/{pregunta.get('puntos_maximos', 0)}")
                                        
                                        if pregunta.get('es_correcta', False):
                                            st.success("✅ Correcta")
                                        else:
                                            st.error("❌ Incorrecta")
                                    
                                    st.write("**Explicación:**")
                                    st.write(pregunta.get('explicacion', 'No disponible'))
                                    
                                    st.write("**Sugerencias:**")
                                    st.write(pregunta.get('sugerencias', 'No disponible'))
                        
                        # Resumen general
                        if 'resumen_general' in correction_data:
                            st.subheader("📋 Resumen general")
                            resumen = correction_data['resumen_general']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Fortalezas:**")
                                for fortaleza in resumen.get('fortalezas', []):
                                    st.write(f"• {fortaleza}")
                            
                            with col2:
                                st.write("**Áreas de mejora:**")
                                for area in resumen.get('areas_mejora', []):
                                    st.write(f"• {area}")
                            
                            st.write("**Recomendaciones:**")
                            for recomendacion in resumen.get('recomendaciones', []):
                                st.write(f"• {recomendacion}")
                        
                        # Información adicional
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'tiempo_estimado_estudio' in correction_data:
                                st.info(f"⏱️ Tiempo estimado de estudio: {correction_data['tiempo_estimado_estudio']}")
                        
                        with col2:
                            if 'recursos_recomendados' in correction_data:
                                st.info("📚 Recursos recomendados:")
                                for recurso in correction_data['recursos_recomendados']:
                                    st.write(f"• {recurso}")
                        
                        # Guardar resultado
                        if corrector.save_exam_result(
                            1, None, uploaded_file.name, subject, 
                            correction_data, ocr_method, text_quality
                        ):
                            st.success("💾 Resultado guardado exitosamente")
                        
                    else:
                        st.error(f"❌ Error en la corrección: {error}")
                else:
                    st.error("❌ No se pudo extraer texto del archivo")
    
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

