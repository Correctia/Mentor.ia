<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corrector de Exámenes con IA</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf-lib/1.17.1/pdf-lib.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        
        .nav-tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            color: #6c757d;
            transition: all 0.3s ease;
        }
        
        .nav-tab.active {
            background: white;
            color: #495057;
            border-bottom: 3px solid #007bff;
        }
        
        .nav-tab:hover {
            background: #e9ecef;
            color: #495057;
        }
        
        .tab-content {
            padding: 30px;
        }
        
        .tab-pane {
            display: none;
        }
        
        .tab-pane.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        
        .form-control {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
        }
        
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,123,255,0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
            color: white;
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(40,167,69,0.4);
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
            color: #212529;
        }
        
        .btn-warning:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255,193,7,0.4);
        }
        
        .file-upload {
            border: 2px dashed #007bff;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover {
            background: #e6f3ff;
            border-color: #0056b3;
        }
        
        .file-upload.dragover {
            background: #cce7ff;
            border-color: #0056b3;
        }
        
        .camera-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .camera-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .camera-preview {
            width: 100%;
            max-width: 500px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .classes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .class-card {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .class-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        
        .class-card h3 {
            color: #495057;
            margin-bottom: 10px;
        }
        
        .class-card p {
            color: #6c757d;
            margin-bottom: 15px;
        }
        
        .exam-item {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .results-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .correction-item {
            background: white;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        
        .correction-item.correct {
            border-left-color: #28a745;
        }
        
        .hidden {
            display: none;
        }
        
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 10px;
            border: 1px solid transparent;
        }
        
        .alert-success {
            color: #155724;
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        
        .alert-error {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        
        .alert-info {
            color: #0c5460;
            background-color: #d1ecf1;
            border-color: #bee5eb;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        @media (max-width: 768px) {
            .nav-tabs {
                flex-direction: column;
            }
            
            .camera-controls {
                flex-direction: column;
            }
            
            .classes-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📝 Corrector de Exámenes con IA</h1>
            <p>Corrección automática inteligente para asignaturas de letras y ciencias</p>
        </div>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('upload')">📁 Subir Examen</button>
            <button class="nav-tab" onclick="showTab('camera')">📷 Escanear</button>
            <button class="nav-tab" onclick="showTab('classes')">🎓 Mis Clases</button>
            <button class="nav-tab" onclick="showTab('results')">📊 Resultados</button>
        </div>
        
        <div class="tab-content">
            <!-- Pestaña de Subir Examen -->
            <div id="upload" class="tab-pane active">
                <h2>📁 Subir Examen</h2>
                
                <div class="form-group">
                    <label for="examSubject">Asignatura</label>
                    <select id="examSubject" class="form-control">
                        <option value="">Selecciona una asignatura</option>
                        <option value="letras">📚 Letras (Lengua, Historia, Filosofía, etc.)</option>
                        <option value="ciencias">🔬 Ciencias (Matemáticas, Física, Química, etc.)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="examClass">Clase</label>
                    <select id="examClass" class="form-control">
                        <option value="">Selecciona una clase</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="examTitle">Título del Examen</label>
                    <input type="text" id="examTitle" class="form-control" placeholder="Ej: Examen de Matemáticas - Tema 5">
                </div>
                
                <div class="form-group">
                    <label for="rubric">Criterios de Corrección (Opcional)</label>
                    <textarea id="rubric" class="form-control" rows="4" placeholder="Especifica los criterios de corrección, puntuación por pregunta, etc."></textarea>
                </div>
                
                <div class="file-upload" id="fileUpload">
                    <div>
                        <h3>📎 Arrastra y suelta archivos aquí</h3>
                        <p>o haz clic para seleccionar</p>
                        <p><small>Formatos soportados: PDF, JPG, PNG, JPEG</small></p>
                    </div>
                    <input type="file" id="fileInput" multiple accept=".pdf,.jpg,.jpeg,.png" style="display: none;">
                </div>
                
                <div id="fileList" class="hidden">
                    <h4>Archivos seleccionados:</h4>
                    <ul id="selectedFiles"></ul>
                </div>
                
                <div class="form-group">
                    <button id="processExam" class="btn btn-primary" onclick="processExam()">🚀 Procesar Examen</button>
                </div>
            </div>
            
            <!-- Pestaña de Cámara -->
            <div id="camera" class="tab-pane">
                <h2>📷 Escanear Examen</h2>
                
                <div class="camera-section">
                    <div class="camera-controls">
                        <button id="startCamera" class="btn btn-primary" onclick="startCamera()">📹 Iniciar Cámara</button>
                        <button id="capturePhoto" class="btn btn-success hidden" onclick="capturePhoto()">📸 Capturar</button>
                        <button id="stopCamera" class="btn btn-warning hidden" onclick="stopCamera()">⏹️ Detener</button>
                    </div>
                    
                    <div id="cameraContainer" class="hidden">
                        <video id="videoElement" class="camera-preview" autoplay></video>
                        <canvas id="canvasElement" class="camera-preview hidden"></canvas>
                    </div>
                </div>
                
                <div id="capturedImages" class="hidden">
                    <h4>Imágenes capturadas:</h4>
                    <div id="imageGallery"></div>
                    <button class="btn btn-primary" onclick="processScannedImages()">🔍 Procesar Imágenes</button>
                </div>
            </div>
            
            <!-- Pestaña de Clases -->
            <div id="classes" class="tab-pane">
                <h2>🎓 Gestión de Clases</h2>
                
                <div class="form-group">
                    <button class="btn btn-primary" onclick="showCreateClassForm()">➕ Crear Nueva Clase</button>
                </div>
                
                <div id="createClassForm" class="hidden">
                    <h3>Crear Nueva Clase</h3>
                    <div class="form-group">
                        <label for="className">Nombre de la Clase</label>
                        <input type="text" id="className" class="form-control" placeholder="Ej: Matemáticas 1º ESO">
                    </div>
                    <div class="form-group">
                        <label for="classDescription">Descripción</label>
                        <textarea id="classDescription" class="form-control" rows="3" placeholder="Descripción de la clase"></textarea>
                    </div>
                    <div class="form-group">
                        <label for="classSubject">Tipo de Asignatura</label>
                        <select id="classSubject" class="form-control">
                            <option value="letras">📚 Letras</option>
                            <option value="ciencias">🔬 Ciencias</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <button class="btn btn-success" onclick="createClass()">✅ Crear Clase</button>
                        <button class="btn btn-warning" onclick="hideCreateClassForm()">❌ Cancelar</button>
                    </div>
                </div>
                
                <div id="classesContainer" class="classes-grid">
                    <!-- Las clases se cargarán aquí dinámicamente -->
                </div>
            </div>
            
            <!-- Pestaña de Resultados -->
            <div id="results" class="tab-pane">
                <h2>📊 Resultados de Corrección</h2>
                
                <div id="resultsContainer">
                    <div class="alert alert-info">
                        <p>📋 Aquí aparecerán los resultados de la corrección una vez que proceses un examen.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Loading overlay -->
        <div id="loadingOverlay" class="loading">
            <div class="spinner"></div>
            <h3>Procesando examen...</h3>
            <p id="loadingMessage">Inicializando...</p>
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill"></div>
            </div>
        </div>
    </div>

    <script>
        // Variables globales
        let currentStream = null;
        let capturedImages = [];
        let classes = JSON.parse(localStorage.getItem('examClasses')) || [];
        let examResults = JSON.parse(localStorage.getItem('examResults')) || [];
        
        // Configuración de APIs (en producción, esto debería estar en el servidor)
        const API_CONFIG = {
            deepseek: {
                baseUrl: 'https://api.deepseek.com/v1',
                apiKey: 'sk-42d24fd956db4146b24782e33879b6ad' // Reemplazar con tu clave real
            },
            googleVision: {
                apiKey: 'AIzaSyAyGT7uDH5Feaqtc27fcF7ArgkrRO8jU0Q' // Reemplazar con tu clave real
            },
            mathpix: {
                appId: 'YOUR_MATHPIX_APP_ID', // Reemplazar con tu app ID
                appKey: 'YOUR_MATHPIX_APP_KEY' // Reemplazar con tu clave real
            }
        };
        
        // Inicialización
        document.addEventListener('DOMContentLoaded', function() {
            loadClasses();
            updateClassSelect();
            setupFileUpload();
            loadResults();
        });
        
        // Gestión de pestañas
        function showTab(tabName) {
            // Ocultar todas las pestañas
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            
            // Desactivar todos los botones
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Mostrar la pestaña seleccionada
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Configuración de carga de archivos
        function setupFileUpload() {
            const fileUpload = document.getElementById('fileUpload');
            const fileInput = document.getElementById('fileInput');
            
            fileUpload.addEventListener('click', () => fileInput.click());
            fileUpload.addEventListener('dragover', (e) => {
                e.preventDefault();
                fileUpload.classList.add('dragover');
            });
            fileUpload.addEventListener('dragleave', () => {
                fileUpload.classList.remove('dragover');
            });
            fileUpload.addEventListener('drop', (e) => {
                e.preventDefault();
                fileUpload.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });
            
            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });
        }
        
        function handleFiles(files) {
            const fileList = document.getElementById('fileList');
            const selectedFiles = document.getElementById('selectedFiles');
            
            selectedFiles.innerHTML = '';
            
            Array.from(files).forEach(file => {
                const li = document.createElement('li');
                li.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                selectedFiles.appendChild(li);
            });
            
            fileList.classList.remove('hidden');
        }
        
        // Funciones de cámara
        function startCamera() {
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
                .then(stream => {
                    currentStream = stream;
                    const video = document.getElementById('videoElement');
                    video.srcObject = stream;
                    
                    document.getElementById('cameraContainer').classList.remove('hidden');
                    document.getElementById('startCamera').classList.add('hidden');
                    document.getElementById('capturePhoto').classList.remove('hidden');
                    document.getElementById('stopCamera').classList.remove('hidden');
                })
                .catch(err => {
                    showAlert('Error al acceder a la cámara: ' + err.message, 'error');
                });
        }
        
        function capturePhoto() {
            const video = document.getElementById('videoElement');
            const canvas = document.getElementById('canvasElement');
            const context = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg');
            capturedImages.push(imageData);
            
            updateImageGallery();
            showAlert('Imagen capturada correctamente', 'success');
        }
        
        function stopCamera() {
            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
                currentStream = null;
            }
            
            document.getElementById('cameraContainer').classList.add('hidden');
            document.getElementById('startCamera').classList.remove('hidden');
            document.getElementById('capturePhoto').classList.add('hidden');
            document.getElementById('stopCamera').classList.add('hidden');
        }
        
        function updateImageGallery() {
            const gallery = document.getElementById('imageGallery');
            gallery.innerHTML = '';
            
            capturedImages.forEach((imageData, index) => {
                const img = document.createElement('img');
                img.src = imageData;
                img.style.width = '150px';
                img.style.height = '200px';
                img.style.objectFit = 'cover';
                img.style.margin = '5px';
                img.style.borderRadius = '5px';
                gallery.appendChild(img);
            });
            
            if (capturedImages.length > 0) {
                document.getElementById('capturedImages').classList.remove('hidden');
            }
        }
        
        // Gestión de clases
        function showCreateClassForm() {
            document.getElementById('createClassForm').classList.remove('hidden');
        }
        
        function hideCreateClassForm() {
            document.getElementById('createClassForm').classList.add('hidden');
        }
        
        function createClass() {
            const name = document.getElementById('className').value;
            const description = document.getElementById('classDescription').value;
            const subject = document.getElementById('classSubject').value;
            
            if (!name.trim()) {
                showAlert('El nombre de la clase es obligatorio', 'error');
                return;
            }
            
            const newClass = {
                id: Date.now(),
                name,
                description,
                subject,
                exams: [],
                createdAt: new Date().toISOString()
            };
            
            classes.push(newClass);
            localStorage.setItem('examClasses', JSON.stringify(classes));
            
            loadClasses();
            updateClassSelect();
            hideCreateClassForm();
            
            // Limpiar formulario
            document.getElementById('className').value = '';
            document.getElementById('classDescription').value = '';
            
            showAlert('Clase creada correctamente', 'success');
        }
        
        function loadClasses() {
            const container = document.getElementById('classesContainer');
            container.innerHTML = '';
            
            if (classes.length === 0) {
                container.innerHTML = '<div class="alert alert-info">No hay clases creadas aún. Crea tu primera clase para empezar.</div>';
                return;
            }
            
            classes.forEach(cls => {
                const card = document.createElement('div');
                card.className = 'class-card';
                card.innerHTML = `
                    <h3>${cls.subject === 'letras' ? '📚' : '🔬'} ${cls.name}</h3>
                    <p>${cls.description || 'Sin descripción'}</p>
                    <div class="exam-stats">
                        <small>Exámenes: ${cls.exams.length}</small>
                    </div>
                    <div style="margin-top: 15px;">
                        <button class="btn btn-primary" onclick="viewClass(${cls.id})">👁️ Ver</button>
                        <button class="btn btn-warning" onclick="deleteClass(${cls.id})">🗑️ Eliminar</button>
                    </div>
                `;
                container.appendChild(card);
            });
        }
        
        function updateClassSelect() {
            const select = document.getElementById('examClass');
            select.innerHTML = '<option value="">Selecciona una clase</option>';
            
            classes.forEach(cls => {
                const option = document.createElement('option');
                option.value = cls.id;
                option.textContent = `${cls.subject === 'letras' ? '📚' : '🔬'} ${cls.name}`;
                select.appendChild(option);
            });
        }
        
        function viewClass(classId) {
            const cls = classes.find(c => c.id === classId);
            if (!cls) return;
            
            // Aquí puedes implementar la vista detallada de la clase
            showAlert(`Viendo clase: ${cls.name}`, 'info');
        }
        
        function deleteClass(classId) {
            if (confirm('¿Estás seguro de que quieres eliminar esta clase?')) {
                classes = classes.filter(c => c.id !== classId);
                localStorage.setItem('examClasses', JSON.stringify(classes));
                loadClasses();
                updateClassSelect();
                showAlert('Clase eliminada correctamente', 'success');
            }
        }
        
        // Procesamiento de exámenes
        async function processExam() {
            const subject = document.getElementById('examSubject').value;
            const classId = document.getElementById('examClass').value;
            const title = document.getElementById('examTitle').value;
            const rubric = document.getElementById('rubric').value;
            const fileInput = document.getElementById('fileInput');
            
            if (!subject || !title || !fileInput.files.length) {
                showAlert('Por favor, completa todos los campos obligatorios y selecciona al menos un archivo', 'error');
                return;
            }
            
            showLoading();
            
            try {
                // Simular procesamiento
                await simulateProcessing();
                
                // Aquí implementarías la lógica real de procesamiento
                const result = await processExamFiles(fileInput.files, subject, rubric);
                
                // Guardar resultado
                const examResult = {
                    id: Date.now(),
                    title,
                    subject,
                    classId,
                    corrections: result.corrections,
                    score: result.score,
                    processedAt: new Date().toISOString(),
                    pdfUrl: result.pdfUrl
                };
                
                examResults.push(examResult);
                localStorage.setItem('examResults', JSON.stringify(examResults));
                
                // Añadir a la clase si se seleccionó una
                if (classId) {
                    const cls = classes.find(c => c.id == classId);
                    if (cls) {
                        cls.exams.push(examResult.id);
                        localStorage.setItem('examClasses', JSON.stringify(classes));
                    }
                }
                
                hideLoading();
                showResults(examResult);
                showTab('results');
                
            } catch (error) {
                hideLoading();
                showAlert('Error al procesar el examen: ' + error.message, 'error');
            }
        }
        
        async function processScannedImages() {
            if (capturedImages.length === 0) {
                showAlert('No hay imágenes capturadas para procesar', 'error');
                return;
            }
            
            showLoading();
            
            try {
                // Simular procesamiento
                await simulateProcessing();
                
                // Aquí implementarías la lógica real de procesamiento de imágenes
                const result = await processImages(capturedImages);
                
                hideLoading();
                showAlert('Imágenes procesadas correctamente', 'success');
                
            } catch (error) {
                hideLoading();
                showAlert('Error al procesar las imágenes: ' + error.message, 'error');
            }
        }
        
        // Funciones auxiliares de procesamiento
        async function processExamFiles(files, subject, rubric) {
            const corrections = [];
            let totalScore = 0;
            
            for (const file of files) {
                if (file.type === 'application/pdf') {
                    // Procesar PDF
                    const text = await extractTextFromPDF(file);
                    const ocrResult = await performOCR(text, subject);
                    const correction = await correctWithAI(ocrResult, subject, rubric);
                    corrections.push(correction);
                }
                totalScore += correction.score;
            }
            
            // Generar PDF corregido
            const pdfUrl = await generateCorrectedPDF(corrections);
            
            return {
                corrections,
                score: totalScore / corrections.length,
                pdfUrl
            };
        }
        
        async function extractTextFromPDF(file) {
            // Simulación de extracción de texto de PDF
            // En producción, usarías una librería como PDF.js o pdf2pic + OCR
            return "Texto extraído del PDF simulado";
        }
        
        async function performOCR(text, subject) {
            // Determinar qué OCR usar según la asignatura
            if (subject === 'letras') {
                return await googleVisionOCR(text);
            } else {
                return await mathpixOCR(text);
            }
        }
        
        async function performOCROnImage(file, subject) {
            const base64 = await fileToBase64(file);
            
            if (subject === 'letras') {
                return await googleVisionOCR(base64);
            } else {
                return await mathpixOCR(base64);
            }
        }
        
        async function googleVisionOCR(imageData) {
            // Simulación de Google Vision API
            // En producción, harías una llamada real a la API
            return {
                text: "Texto extraído con Google Vision (simulado)",
                confidence: 0.95
            };
        }
        
        async function mathpixOCR(imageData) {
            // Simulación de Mathpix API
            // En producción, harías una llamada real a la API
            return {
                text: "Ecuaciones y texto matemático extraído con Mathpix (simulado)",
                latex: "\\int_{0}^{1} x^2 dx = \\frac{1}{3}",
                confidence: 0.92
            };
        }
        
        async function correctWithAI(ocrResult, subject, rubric) {
            // Simulación de corrección con DeepSeek
            // En producción, harías una llamada real a la API de DeepSeek
            
            const prompt = `
                Corrige el siguiente examen de ${subject}:
                
                Texto del examen:
                ${ocrResult.text}
                
                Criterios de corrección:
                ${rubric || 'Criterios estándar'}
                
                Por favor, proporciona:
                1. Respuestas correctas
                2. Errores identificados
                3. Puntuación
                4. Comentarios de mejora
            `;
            
            // Simulación de respuesta de IA
            return {
                originalText: ocrResult.text,
                errors: [
                    {
                        question: "Pregunta 1",
                        studentAnswer: "Respuesta incorrecta del estudiante",
                        correctAnswer: "Respuesta correcta",
                        feedback: "La respuesta no es correcta porque...",
                        points: 0,
                        maxPoints: 2
                    },
                    {
                        question: "Pregunta 2", 
                        studentAnswer: "Respuesta parcialmente correcta",
                        correctAnswer: "Respuesta completa correcta",
                        feedback: "La respuesta está bien encaminada pero falta...",
                        points: 1,
                        maxPoints: 2
                    }
                ],
                score: 7.5,
                maxScore: 10,
                generalFeedback: "El estudiante muestra comprensión básica pero necesita mejorar en..."
            };
        }
        
        async function generateCorrectedPDF(corrections) {
            // Generar PDF corregido usando jsPDF
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            let yPosition = 20;
            
            // Título
            doc.setFontSize(16);
            doc.text('Examen Corregido', 20, yPosition);
            yPosition += 20;
            
            // Procesar cada corrección
            corrections.forEach((correction, index) => {
                doc.setFontSize(12);
                doc.text(`Página ${index + 1}`, 20, yPosition);
                yPosition += 10;
                
                // Mostrar errores
                correction.errors.forEach((error, errorIndex) => {
                    doc.setTextColor(255, 0, 0); // Rojo para errores
                    doc.text(`❌ ${error.question}`, 20, yPosition);
                    yPosition += 7;
                    
                    doc.setTextColor(0, 0, 0); // Negro para texto normal
                    doc.text(`Respuesta: ${error.studentAnswer}`, 30, yPosition);
                    yPosition += 7;
                    
                    doc.setTextColor(0, 150, 0); // Verde para respuesta correcta
                    doc.text(`Correcto: ${error.correctAnswer}`, 30, yPosition);
                    yPosition += 7;
                    
                    doc.setTextColor(255, 0, 0); // Rojo para feedback
                    doc.text(`Comentario: ${error.feedback}`, 30, yPosition);
                    yPosition += 7;
                    
                    doc.setTextColor(0, 0, 0);
                    doc.text(`Puntos: ${error.points}/${error.maxPoints}`, 30, yPosition);
                    yPosition += 15;
                    
                    // Nueva página si es necesario
                    if (yPosition > 250) {
                        doc.addPage();
                        yPosition = 20;
                    }
                });
                
                // Puntuación total
                doc.setFontSize(14);
                doc.setTextColor(0, 0, 255); // Azul para puntuación
                doc.text(`Puntuación total: ${correction.score}/${correction.maxScore}`, 20, yPosition);
                yPosition += 20;
            });
            
            // Generar URL del PDF
            const pdfBlob = doc.output('blob');
            const pdfUrl = URL.createObjectURL(pdfBlob);
            
            return pdfUrl;
        }
        
        async function processImages(images) {
            // Procesar las imágenes capturadas
            const results = [];
            
            for (const imageData of images) {
                const ocrResult = await performOCROnImage(dataURLtoBlob(imageData), 'letras');
                const correction = await correctWithAI(ocrResult, 'letras', '');
                results.push(correction);
            }
            
            return results;
        }
        
        // Funciones auxiliares
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }
        
        function dataURLtoBlob(dataURL) {
            const arr = dataURL.split(',');
            const mime = arr[0].match(/:(.*?);/)[1];
            const bstr = atob(arr[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while (n--) {
                u8arr[n] = bstr.charCodeAt(n);
            }
            return new Blob([u8arr], { type: mime });
        }
        
        // Mostrar resultados
        function showResults(examResult) {
            const container = document.getElementById('resultsContainer');
            container.innerHTML = '';
            
            // Información general
            const header = document.createElement('div');
            header.className = 'alert alert-info';
            header.innerHTML = `
                <h3>📋 ${examResult.title}</h3>
                <p><strong>Asignatura:</strong> ${examResult.subject === 'letras' ? '📚 Letras' : '🔬 Ciencias'}</p>
                <p><strong>Fecha:</strong> ${new Date(examResult.processedAt).toLocaleString()}</p>
                <p><strong>Puntuación:</strong> ${examResult.score.toFixed(1)}/10</p>
            `;
            container.appendChild(header);
            
            // Botón de descarga
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'btn btn-success';
            downloadBtn.innerHTML = '📥 Descargar PDF Corregido';
            downloadBtn.onclick = () => {
                const link = document.createElement('a');
                link.href = examResult.pdfUrl;
                link.download = `${examResult.title}_corregido.pdf`;
                link.click();
            };
            container.appendChild(downloadBtn);
            
            // Resultados de corrección
            const resultsDiv = document.createElement('div');
            resultsDiv.className = 'results-section';
            resultsDiv.innerHTML = '<h4>🔍 Detalle de Correcciones</h4>';
            
            examResult.corrections.forEach((correction, index) => {
                const correctionDiv = document.createElement('div');
                correctionDiv.innerHTML = `
                    <h5>Página ${index + 1}</h5>
                    <p><strong>Puntuación:</strong> ${correction.score}/${correction.maxScore}</p>
                    <p><strong>Comentario general:</strong> ${correction.generalFeedback}</p>
                `;
                
                correction.errors.forEach(error => {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = error.points === error.maxPoints ? 'correction-item correct' : 'correction-item';
                    errorDiv.innerHTML = `
                        <h6>${error.question}</h6>
                        <p><strong>Respuesta del estudiante:</strong> ${error.studentAnswer}</p>
                        <p><strong>Respuesta correcta:</strong> ${error.correctAnswer}</p>
                        <p><strong>Comentario:</strong> ${error.feedback}</p>
                        <p><strong>Puntos:</strong> ${error.points}/${error.maxPoints}</p>
                    `;
                    correctionDiv.appendChild(errorDiv);
                });
                
                resultsDiv.appendChild(correctionDiv);
            });
            
            container.appendChild(resultsDiv);
        }
        
        function loadResults() {
            // Cargar resultados previos si existen
            if (examResults.length > 0) {
                // Mostrar el último resultado
                const lastResult = examResults[examResults.length - 1];
                showResults(lastResult);
            }
        }
        
        // Funciones de UI
        function showLoading() {
            document.getElementById('loadingOverlay').classList.add('active');
        }
        
        function hideLoading() {
            document.getElementById('loadingOverlay').classList.remove('active');
        }
        
        function showAlert(message, type = 'info') {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;
            
            // Insertar al principio del contenido activo
            const activeTab = document.querySelector('.tab-pane.active');
            activeTab.insertBefore(alertDiv, activeTab.firstChild);
            
            // Eliminar después de 5 segundos
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
        
        async function simulateProcessing() {
            const messages = [
                'Extrayendo texto del documento...',
                'Analizando contenido con OCR...',
                'Procesando con IA...',
                'Generando correcciones...',
                'Creando PDF corregido...'
            ];
            
            const progressFill = document.getElementById('progressFill');
            const loadingMessage = document.getElementById('loadingMessage');
            
            for (let i = 0; i < messages.length; i++) {
                loadingMessage.textContent = messages[i];
                progressFill.style.width = `${((i + 1) / messages.length) * 100}%`;
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        // Funciones de API reales (para implementar en producción)
        
        /*
        // Ejemplo de implementación real con DeepSeek
        async function callDeepSeekAPI(prompt) {
            const response = await fetch(`${API_CONFIG.deepseek.baseUrl}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_CONFIG.deepseek.apiKey}`
                },
                body: JSON.stringify({
                    model: 'deepseek-chat',
                    messages: [
                        {
                            role: 'system',
                            content: 'Eres un profesor experto en corrección de exámenes. Analiza el examen y proporciona correcciones detalladas.'
                        },
                        {
                            role: 'user',
                            content: prompt
                        }
                    ],
                    max_tokens: 4000,
                    temperature: 0.1
                })
            });
            
            return await response.json();
        }
        
        // Ejemplo de implementación real con Google Vision
        async function callGoogleVisionAPI(base64Image) {
            const response = await fetch(`https://vision.googleapis.com/v1/images:annotate?key=${API_CONFIG.googleVision.apiKey}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    requests: [{
                        image: {
                            content: base64Image.split(',')[1]
                        },
                        features: [
                            {
                                type: 'TEXT_DETECTION',
                                maxResults: 50
                            }
                        ]
                    }]
                })
            });
            
            return await response.json();
        }
        
        // Ejemplo de implementación real con Mathpix
        async function callMathpixAPI(base64Image) {
            const response = await fetch('https://api.mathpix.com/v3/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'app_id': API_CONFIG.mathpix.appId,
                    'app_key': API_CONFIG.mathpix.appKey
                },
                body: JSON.stringify({
                    src: base64Image,
                    formats: ['text', 'latex_styled'],
                    data_options: {
                        include_asciimath: true,
                        include_latex: true
                    }
                })
            });
            
            return await response.json();
        }
        */
    </script>
</body>
</html> subject, rubric);
                    corrections.push(correction);
                } else {
                    // Procesar imagen
                    const ocrResult = await performOCROnImage(file, subject);
                    const correction = await correctWithAI(ocrResult,
