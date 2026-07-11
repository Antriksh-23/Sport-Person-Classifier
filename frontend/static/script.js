document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const modeUpload = document.getElementById('mode-upload');
    const modeWebcam = document.getElementById('mode-webcam');
    const uploadZone = document.getElementById('upload-zone');
    const webcamZone = document.getElementById('webcam-zone');
    const previewZone = document.getElementById('preview-zone');
    const resultZone = document.getElementById('result-zone');
    
    const fileUpload = document.getElementById('file-upload');
    const imagePreview = document.getElementById('image-preview');
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const captureBtn = document.getElementById('capture-btn');
    const classifyBtn = document.getElementById('classify-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const laserScanner = document.getElementById('laser-scanner');
    const processingState = document.getElementById('processing-state');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    let base64Image = null;
    let stream = null;

    /* ==================================================
       0. MOBILE MENU TOGGLE
       ================================================== */
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const navLinks = document.getElementById('nav-links');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
        
        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
            });
        });
    }

    /* ==================================================
       1. MODE TOGGLING (UPLOAD VS WEBCAM)
       ================================================== */
    modeUpload.addEventListener('click', () => {
        modeUpload.classList.add('active');
        modeWebcam.classList.remove('active');
        uploadZone.classList.remove('hidden');
        webcamZone.classList.add('hidden');
        stopWebcam();
        resetState();
    });

    modeWebcam.addEventListener('click', () => {
        modeWebcam.classList.add('active');
        modeUpload.classList.remove('active');
        webcamZone.classList.remove('hidden');
        uploadZone.classList.add('hidden');
        resetState();
        startWebcam();
    });

    /* ==================================================
       2. WEBCAM LOGIC
       ================================================== */
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            webcamVideo.srcObject = stream;
        } catch (err) {
            showError("Camera access denied or unavailable.");
            modeUpload.click(); // fallback to upload
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamVideo.srcObject = null;
        }
    }

    captureBtn.addEventListener('click', () => {
        if (!stream) return;
        
        // Match canvas to video dimensions
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        
        // Draw frame to canvas
        const ctx = webcamCanvas.getContext('2d');
        // If the video is mirrored via CSS, we should flip the context so the output image looks correct
        ctx.translate(webcamCanvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0);
        
        // Extract base64
        base64Image = webcamCanvas.toDataURL('image/jpeg', 0.9);
        imagePreview.src = base64Image;
        
        // UI Switch
        webcamZone.classList.add('hidden');
        previewZone.classList.remove('hidden');
    });

    /* ==================================================
       3. FILE UPLOAD LOGIC (DRAG & DROP)
       ================================================== */
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    uploadZone.addEventListener('drop', handleDrop, false);
    uploadZone.addEventListener('click', () => fileUpload.click());
    fileUpload.addEventListener('change', handleChange, false);

    function handleDrop(e) {
        const files = e.dataTransfer.files;
        handleFiles(files);
    }

    function handleChange(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            showError("Please upload a valid image file (JPEG, PNG).");
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            base64Image = reader.result;
            imagePreview.src = base64Image;
            
            uploadZone.classList.add('hidden');
            previewZone.classList.remove('hidden');
        };
    }

    /* ==================================================
       4. PREDICTION LOGIC
       ================================================== */
    resetBtn.addEventListener('click', () => {
        resetState();
        if (modeWebcam.classList.contains('active')) {
            webcamZone.classList.remove('hidden');
        } else {
            uploadZone.classList.remove('hidden');
        }
    });

    function resetState() {
        base64Image = null;
        imagePreview.src = '';
        previewZone.classList.add('hidden');
        resultZone.classList.add('hidden');
        errorMessage.classList.add('hidden');
        fileUpload.value = '';
    }

    classifyBtn.addEventListener('click', async () => {
        if (!base64Image) return;

        // Start processing UI (Laser scanner)
        laserScanner.classList.remove('hidden');
        processingState.classList.remove('hidden');
        classifyBtn.disabled = true;
        errorMessage.classList.add('hidden');
        resultZone.classList.add('hidden');
        
        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: base64Image })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }
            
            const endTime = performance.now();
            const processTime = ((endTime - startTime) / 1000).toFixed(2);
            document.getElementById('timestamp').textContent = `Processed in ${processTime}s`;

            displayResult(data);
        } catch (error) {
            showError(error.message);
        } finally {
            laserScanner.classList.add('hidden');
            processingState.classList.add('hidden');
            classifyBtn.disabled = false;
        }
    });

    function displayResult(data) {
        document.getElementById('predicted-athlete').textContent = formatName(data.athlete);
        
        const confSpan = document.getElementById('confidence-score');
        const confBar = document.getElementById('confidence-bar');
        
        confSpan.textContent = data.confidence + '%';
        // Reset bar width for animation
        confBar.style.width = '0%';
        setTimeout(() => {
            confBar.style.width = data.confidence + '%';
        }, 50);

        // Populate Top Probabilities
        const grid = document.getElementById('probability-grid');
        grid.innerHTML = '';
        
        // Sort descending and take top 4
        const sortedProbs = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4);
            
        sortedProbs.forEach(([athlete, prob]) => {
            const div = document.createElement('div');
            div.className = 'prob-item';
            div.innerHTML = `
                <span style="color: var(--text-muted);">${formatName(athlete)}</span>
                <span style="color: var(--text-main); font-weight: 600;">${prob}%</span>
            `;
            grid.appendChild(div);
        });

        resultZone.classList.remove('hidden');
    }

    function formatName(name) {
        if (!name) return "Unknown";
        return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    /* ==================================================
       5. MICRO-INTERACTIONS & ANIMATIONS
       ================================================== */
       
    // Scroll Intersection Observer for Fade-Up
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.observe-me').forEach(el => observer.observe(el));

    // 3D Tilt Effect on Cards
    document.querySelectorAll('.tilt-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = ((y - centerY) / centerY) * -10; // max 10 deg
            const rotateY = ((x - centerX) / centerX) * 10;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
        });
    });

    // Magnetic Buttons
    document.querySelectorAll('.magnetic').forEach(btn => {
        btn.addEventListener('mousemove', (e) => {
            const rect = btn.getBoundingClientRect();
            const x = (e.clientX - rect.left - rect.width / 2) * 0.2; // 20% pull
            const y = (e.clientY - rect.top - rect.height / 2) * 0.2;
            btn.style.transform = `translate(${x}px, ${y}px)`;
        });
        
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'translate(0px, 0px)';
        });
    });
});
