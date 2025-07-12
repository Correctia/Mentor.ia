import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Tuple, Optional

class AdvancedImageProcessor:
    """Procesador avanzado de imágenes para mejorar la calidad del OCR"""
    
    def __init__(self):
        self.denoise_strength = 10
        self.sharpen_factor = 1.5
        self.contrast_factor = 1.2
        
    def preprocess_image_for_ocr(self, image_bytes: bytes) -> bytes:
        """
        Preprocesa una imagen para mejorar los resultados del OCR
        
        Args:
            image_bytes: Imagen en formato bytes
            
        Returns:
            bytes: Imagen procesada en formato bytes
        """
        # Convertir bytes a imagen PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a escala de grises si es necesario
        if image.mode != 'L':
            image = image.convert('L')
        
        # Aplicar mejoras
        image = self._enhance_contrast(image)
        image = self._sharpen_image(image)
        image = self._remove_noise(image)
        image = self._binarize_image(image)
        
        # Convertir de vuelta a bytes
        output_bytes = io.BytesIO()
        image.save(output_bytes, format='PNG')
        return output_bytes.getvalue()
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Mejora el contraste de la imagen"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(self.contrast_factor)
    
    def _sharpen_image(self, image: Image.Image) -> Image.Image:
        """Aplica un filtro de enfoque a la imagen"""
        return image.filter(ImageFilter.UnsharpMask(
            radius=2, percent=150, threshold=3
        ))
    
    def _remove_noise(self, image: Image.Image) -> Image.Image:
        """Elimina ruido de la imagen"""
        # Convertir a array numpy para usar OpenCV
        img_array = np.array(image)
        
        # Aplicar filtro de desenfoque para reducir ruido
        denoised = cv2.fastNlMeansDenoising(img_array, None, self.denoise_strength, 7, 21)
        
        # Convertir de vuelta a PIL
        return Image.fromarray(denoised)
    
    def _binarize_image(self, image: Image.Image) -> Image.Image:
        """Convierte la imagen a binaria usando umbralización adaptativa"""
        # Convertir a array numpy
        img_array = np.array(image)
        
        # Aplicar umbralización adaptativa
        binary = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(binary)

def preprocess_image(image: Image.Image) -> bytes:
    """
    Función de conveniencia para preprocesar una imagen PIL
    
    Args:
        image: Imagen PIL
        
    Returns:
        bytes: Imagen procesada en formato bytes
    """
    processor = AdvancedImageProcessor()
    
    # Convertir PIL Image a bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Procesar la imagen
    processed_bytes = processor.preprocess_image_for_ocr(img_bytes)
    
    return processed_bytes

def process_captured_image_enhanced(image_bytes: bytes) -> bytes:
    """
    Procesa una imagen capturada con técnicas avanzadas
    
    Args:
        image_bytes: Imagen en formato bytes
        
    Returns:
        bytes: Imagen procesada
    """
    processor = AdvancedImageProcessor()
    return processor.preprocess_image_for_ocr(image_bytes)
