// static/js/main.js

// AI Accessibility Assistant - Main JavaScript

class AccessibilityAssistant {
    constructor() {
        this.currentFeature = 'image-to-speech';
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.currentAudio = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupThemeToggle();
        this.setupAccessibility();
        
        // Show initial feature
        this.showFeature('image-to-speech');
    }
    
    setupEventListeners() {
        // Navigation buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const feature = e.currentTarget.dataset.feature;
                this.showFeature(feature);
            });
        });
        
        // Image upload
        this.setupImageUpload();
        
        // Text to speech
        document.getElementById('text-to-speech-btn').addEventListener('click', () => {
            this.convertTextToSpeech();
        });
        
        // Speech to sign
        document.getElementById('record-btn').addEventListener('click', () => {
            this.toggleRecording();
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardNavigation(e);
        });
    }
    
    setupThemeToggle() {
        const themeToggle = document.getElementById('theme-toggle');
        const themeIcon = document.getElementById('theme-icon');
        
        // Load saved theme or default to light
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
        
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            this.updateThemeIcon(newTheme);
            
            // Announce theme change for screen readers
            this.announceToScreenReader(`Switched to ${newTheme} theme`);
        });
    }
    
    updateThemeIcon(theme) {
        const themeIcon = document.getElementById('theme-icon');
        themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
    
    setupAccessibility() {
        // Add ARIA live regions for dynamic content
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        liveRegion.id = 'live-region';
        document.body.appendChild(liveRegion);
        
        // Skip to main content link
        const skipLink = document.createElement('a');
        skipLink.href = '#main';
        skipLink.textContent = 'Skip to main content';
        skipLink.className = 'sr-only';
        skipLink.style.position = 'absolute';
        skipLink.style.top = '-40px';
        skipLink.style.left = '6px';
        skipLink.style.background = 'var(--accent-color)';
        skipLink.style.color = 'white';
        skipLink.style.padding = '8px';
        skipLink.style.textDecoration = 'none';
        skipLink.style.borderRadius = '4px';
        skipLink.style.zIndex = '10000';
        
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });
        
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });
        
        document.body.insertBefore(skipLink, document.body.firstChild);
    }
    
    handleKeyboardNavigation(e) {
        // Tab navigation enhancement
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
        
        // Escape key to close overlays
        if (e.key === 'Escape') {
            this.hideLoading();
            this.stopAudio();
        }
        
        // Arrow keys for feature navigation
        if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
            e.preventDefault();
            this.navigateFeatures(e.key === 'ArrowRight' ? 1 : -1);
        }
    }
    
    navigateFeatures(direction) {
        const features = ['image-to-speech', 'text-to-speech', 'speech-to-sign'];
        const currentIndex = features.indexOf(this.currentFeature);
        const newIndex = (currentIndex + direction + features.length) % features.length;
        this.showFeature(features[newIndex]);
    }
    
    showFeature(featureName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.feature === featureName) {
                btn.classList.add('active');
            }
        });
        
        // Update sections
        document.querySelectorAll('.feature-section').forEach(section => {
            section.classList.remove('active');
        });
        
        document.getElementById(featureName).classList.add('active');
        this.currentFeature = featureName;
        
        // Announce feature change
        const featureNames = {
            'image-to-speech': 'Image to Speech',
            'text-to-speech': 'Text to Speech',
            'speech-to-sign': 'Speech to Sign Language'
        };
        this.announceToScreenReader(`Switched to ${featureNames[featureName]} feature`);
    }
    
    setupImageUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('image-input');
        
        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // File selection
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleImageUpload(files[0]);
            }
        });
    }
    
    async handleImageUpload(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'application/pdf'];
        
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please upload a valid image file or PDF.');
            return;
        }
        
        if (file.size > 16 * 1024 * 1024) { // 16MB
            this.showError('File size must be less than 16MB.');
            return;
        }
        
        this.showLoading('Processing image...');
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('language', document.getElementById('image-language').value);
        
        try {
            const response = await fetch('/api/image-to-speech', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayImageResult(data);
                this.announceToScreenReader('Image processed successfully. Text extracted and audio generated.');
            } else {
                this.showError(data.error || 'Failed to process image.');
            }
        } catch (error) {
            console.error('Image upload error:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    displayImageResult(data) {
        const resultArea = document.getElementById('image-result');
        
        resultArea.innerHTML = `
            <div class="result-text">
                <h4><i class="fas fa-file-text"></i> Extracted Text:</h4>
                <p>${this.escapeHtml(data.text)}</p>
            </div>
            <div class="audio-controls">
                <button onclick="assistant.playAudio('${data.audio_data}')" aria-label="Play extracted text as speech">
                    <i class="fas fa-play"></i> Play Audio
                </button>
                <button onclick="assistant.downloadAudio('${data.audio_data}', 'extracted-text.mp3')" aria-label="Download audio file">
                    <i class="fas fa-download"></i> Download Audio
                </button>
            </div>
        `;
        
        resultArea.classList.add('show');
    }
    
    async convertTextToSpeech() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to convert.');
            textInput.focus();
            return;
        }
        
        const language = document.getElementById('text-language').value;
        const summarize = document.getElementById('summarize-toggle').checked;
        
        this.showLoading('Converting text to speech...');
        
        try {
            const response = await fetch('/api/text-to-speech', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    language: language,
                    summarize: summarize
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayTextResult(data);
                this.announceToScreenReader('Text converted to speech successfully.');
            } else {
                this.showError(data.error || 'Failed to convert text to speech.');
            }
        } catch (error) {
            console.error('Text to speech error:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    displayTextResult(data) {
        const resultArea = document.getElementById('text-result');
        
        let content = '';
        
        if (data.was_summarized) {
            content += `
                <div class="result-summary">
                    <h4><i class="fas fa-compress-alt"></i> Summary:</h4>
                    <p>${this.escapeHtml(data.processed_text)}</p>
                </div>
            `;
        }
        
        content += `
            <div class="audio-controls">
                <button onclick="assistant.playAudio('${data.audio_data}')" aria-label="Play ${data.was_summarized ? 'summarized' : 'original'} text as speech">
                    <i class="fas fa-play"></i> Play Audio
                </button>
                <button onclick="assistant.downloadAudio('${data.audio_data}', 'text-to-speech.mp3')" aria-label="Download audio file">
                    <i class="fas fa-download"></i> Download Audio
                </button>
            </div>
        `;
        
        resultArea.innerHTML = content;
        resultArea.classList.add('show');
    }
    
    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processSpeechRecording();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            this.updateRecordingUI(true);
            this.announceToScreenReader('Recording started. Speak now.');
            
        } catch (error) {
            console.error('Recording error:', error);
            this.showError('Unable to access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            
            // Stop all tracks
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            this.isRecording = false;
            this.updateRecordingUI(false);
            this.announceToScreenReader('Recording stopped. Processing speech...');
        }
    }
    
    updateRecordingUI(recording) {
        const recordBtn = document.getElementById('record-btn');
        const recordingStatus = document.getElementById('recording-status');
        
        if (recording) {
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<i class="fas fa-stop"></i><span>Stop Recording</span>';
            recordingStatus.textContent = 'Recording... Speak clearly into your microphone.';
        } else {
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Click to Record</span>';
            recordingStatus.textContent = 'Processing your speech...';
        }
    }
    
    async processSpeechRecording() {
        if (this.audioChunks.length === 0) {
            this.showError('No audio recorded. Please try again.');
            return;
        }
        
        this.showLoading('Processing speech...');
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        try {
            const response = await fetch('/api/speech-to-sign', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displaySpeechResult(data);
                this.announceToScreenReader(`Speech recognized: ${data.recognized_text}. Sign language generated.`);
            } else {
                this.showError(data.error || 'Failed to process speech.');
            }
        } catch (error) {
            console.error('Speech processing error:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.hideLoading();
            document.getElementById('recording-status').textContent = '';
        }
    }
    
    displaySpeechResult(data) {
        const resultArea = document.getElementById('speech-result');
        const signArea = document.getElementById('sign-animation');
        
        resultArea.innerHTML = `
            <div class="result-text">
                <h4><i class="fas fa-microphone"></i> Recognized Speech:</h4>
                <p>${this.escapeHtml(data.recognized_text)}</p>
            </div>
        `;
        
        resultArea.classList.add('show');
        
        // Display sign language sequence
        this.displaySignLanguage(data.sign_data, signArea);
    }
    
    displaySignLanguage(signData, container) {
        container.innerHTML = '<h4><i class="fas fa-hands"></i> Sign Language Translation:</h4>';
        
        signData.sequence.forEach((sign, index) => {
            setTimeout(() => {
                const signElement = document.createElement('div');
                signElement.className = 'sign-item';
                signElement.innerHTML = `
                    <div class="sign-gesture">${sign.gesture.replace(/_/g, ' ').toUpperCase()}</div>
                    <div class="sign-description">${this.escapeHtml(sign.description)}</div>
                `;
                container.appendChild(signElement);
                
                // Announce each sign for screen readers
                this.announceToScreenReader(`Sign ${index + 1}: ${sign.description}`);
            }, index * 1000);
        });
        
        container.classList.add('show');
    }
    
    playAudio(audioData) {
        try {
            // Stop any currently playing audio
            this.stopAudio();
            
            // Create blob from base64 data
            const audioBlob = this.base64ToBlob(audioData, 'audio/mp3');
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Create and play audio
            this.currentAudio = new Audio(audioUrl);
            this.currentAudio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
            };
            
            this.currentAudio.onerror = () => {
                this.showError('Error playing audio. Please try again.');
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
            };
            
            this.currentAudio.play();
            this.announceToScreenReader('Audio playback started.');
            
        } catch (error) {
            console.error('Audio playback error:', error);
            this.showError('Error playing audio. Please try again.');
        }
    }
    
    stopAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
            this.announceToScreenReader('Audio playback stopped.');
        }
    }
    
    downloadAudio(audioData, filename) {
        try {
            const audioBlob = this.base64ToBlob(audioData, 'audio/mp3');
            const url = URL.createObjectURL(audioBlob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            URL.revokeObjectURL(url);
            this.announceToScreenReader(`Audio file ${filename} downloaded.`);
            
        } catch (error) {
            console.error('Download error:', error);
            this.showError('Error downloading audio. Please try again.');
        }
    }
    
    base64ToBlob(base64Data, contentType) {
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: contentType });
    }
    
    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const spinner = overlay.querySelector('.loading-spinner p');
        spinner.textContent = message;
        overlay.classList.add('show');
        
        // Announce loading for screen readers
        this.announceToScreenReader(message);
    }
    
    hideLoading() {
        document.getElementById('loading-overlay').classList.remove('show');
    }
    
    showError(message) {
        // Remove any existing error messages
        document.querySelectorAll('.error-message').forEach(el => el.remove());
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${this.escapeHtml(message)}`;
        
        // Insert error message in the current feature section
        const currentSection = document.querySelector('.feature-section.active .glass-card');
        currentSection.insertBefore(errorDiv, currentSection.firstChild);
        
        // Remove error message after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
        
        // Announce error for screen readers
        this.announceToScreenReader(`Error: ${message}`);
    }
    
    showSuccess(message) {
        // Remove any existing success messages
        document.querySelectorAll('.success-message').forEach(el => el.remove());
        
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${this.escapeHtml(message)}`;
        
        // Insert success message in the current feature section
        const currentSection = document.querySelector('.feature-section.active .glass-card');
        currentSection.insertBefore(successDiv, currentSection.firstChild);
        
        // Remove success message after 3 seconds
        setTimeout(() => {
            successDiv.remove();
        }, 3000);
        
        // Announce success for screen readers
        this.announceToScreenReader(message);
    }
    
    announceToScreenReader(message) {
        const liveRegion = document.getElementById('live-region');
        if (liveRegion) {
            liveRegion.textContent = message;
            
            // Clear the message after a short delay
            setTimeout(() => {
                liveRegion.textContent = '';
            }, 1000);
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.assistant = new AccessibilityAssistant();
});

// Handle mouse clicks to remove keyboard navigation styling
document.addEventListener('mousedown', () => {
    document.body.classList.remove('keyboard-navigation');
});

// Service Worker registration for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}