class PotatoDiseaseDetector {
    constructor() {
        this.currentMethod = 'upload';
        this.stream = null;
        this.selectedFile = null;
        this.initializeElements();
        this.checkBrowserCompatibility();
        this.bindEvents();
    }

    checkBrowserCompatibility() {
        // Check for File System Access API support
        this.folderSelectionSupported = 'showSaveFilePicker' in window;
        
        // Update download help text based on browser compatibility
        const downloadHelp = document.getElementById('downloadHelp');
        if (downloadHelp) {
            if (this.folderSelectionSupported) {
                downloadHelp.innerHTML = '✅ Folder selection supported - Choose where to save your PDF!';
                downloadHelp.style.color = '#28a745';
            } else {
                downloadHelp.innerHTML = '📁 Will download to your default Downloads folder';
                downloadHelp.style.color = '#6c757d';
            }
        }
        
        // Update button text based on compatibility
        const downloadBtn = document.getElementById('downloadResultBtn');
        if (downloadBtn && this.folderSelectionSupported) {
            downloadBtn.innerHTML = '<i class="fas fa-folder-open"></i> Choose Folder & Download PDF';
        } else if (downloadBtn) {
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download PDF Report';
        }
        
        console.log('Browser compatibility:', {
            folderSelection: this.folderSelectionSupported,
            userAgent: navigator.userAgent
        });
    }

    initializeElements() {
        // Method cards
        this.uploadCard = document.getElementById('uploadCard');
        this.cameraCard = document.getElementById('cameraCard');
        
        // Sections
        this.uploadSection = document.getElementById('uploadSection');
        this.cameraSection = document.getElementById('cameraSection');
        
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        
        // Camera elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startCameraBtn = document.getElementById('startCamera');
        this.captureBtn = document.getElementById('captureBtn');
        this.stopCameraBtn = document.getElementById('stopCamera');
        
        // Preview and actions
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.predictBtn = document.getElementById('predictBtn');
        this.clearBtn = document.getElementById('clearBtn');
        
        // Results
        this.resultsSection = document.getElementById('resultsSection');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.newAnalysisBtn = document.getElementById('newAnalysisBtn');
        this.downloadResultBtn = document.getElementById('downloadResultBtn');
        
        // Analyzed image display
        this.analyzedImageSection = document.getElementById('analyzedImageSection');
        this.analyzedImage = document.getElementById('analyzedImage');
        
        // Result elements
        this.diseaseName = document.getElementById('diseaseName');
        this.diseaseDescription = document.getElementById('diseaseDescription');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.confidenceBadge = document.getElementById('confidenceBadge');
        this.diseaseIcon = document.getElementById('diseaseIcon');
        this.timestamp = document.getElementById('timestamp');
        this.probabilities = document.getElementById('probabilities');
        this.recommendationList = document.getElementById('recommendationList');
    }

    bindEvents() {
        // Method switching
        this.uploadCard.addEventListener('click', () => this.switchMethod('upload'));
        this.cameraCard.addEventListener('click', () => this.switchMethod('camera'));
        
        // Upload events with touch support
        this.uploadArea.addEventListener('click', (e) => {
            console.log('Upload area clicked');
            this.fileInput.click();
        });
        
        // Touch events for mobile
        this.uploadArea.addEventListener('touchend', (e) => {
            e.preventDefault();
            console.log('Upload area touched');
            this.fileInput.click();
        });
        
        // Specific handler for browse text
        const browseText = document.querySelector('.browse-text');
        if (browseText) {
            browseText.addEventListener('click', (e) => {
                e.stopPropagation();
                console.log('Browse text clicked');
                this.fileInput.click();
            });
            
            browseText.addEventListener('touchend', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Browse text touched');
                this.fileInput.click();
            });
        }
        
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.fileInput.addEventListener('change', (e) => {
            console.log('File input changed:', e.target.files);
            this.handleFileSelect(e);
        });
        
        // Camera events
        this.startCameraBtn.addEventListener('click', this.startCamera.bind(this));
        this.captureBtn.addEventListener('click', this.capturePhoto.bind(this));
        this.stopCameraBtn.addEventListener('click', this.stopCamera.bind(this));
        
        // Action buttons
        this.predictBtn.addEventListener('click', this.makePrediction.bind(this));
        this.clearBtn.addEventListener('click', this.clearSelection.bind(this));
        this.newAnalysisBtn.addEventListener('click', this.newAnalysis.bind(this));
        this.downloadResultBtn.addEventListener('click', this.downloadReport.bind(this));
    }

    switchMethod(method) {
        this.currentMethod = method;
        
        // Update card states
        this.uploadCard.classList.toggle('active', method === 'upload');
        this.cameraCard.classList.toggle('active', method === 'camera');
        
        // Show/hide sections
        this.uploadSection.style.display = method === 'upload' ? 'block' : 'none';
        this.cameraSection.style.display = method === 'camera' ? 'block' : 'none';
        
        // Stop camera if switching away
        if (method !== 'camera') {
            this.stopCamera();
        }
        
        // Clear any existing selections
        this.clearSelection();
    }

    // Upload handling
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        console.log('Processing file:', file.name, file.type, file.size);
        
        if (!this.isValidImageFile(file)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG)');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            return;
        }

        this.selectedFile = file;
        console.log('File selected successfully:', file.name);
        this.displayImagePreview(file);
    }

    isValidImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        return validTypes.includes(file.type);
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.imagePreview.style.display = 'block';
            this.imagePreview.classList.add('fade-in');
            this.hideResults();
        };
        reader.readAsDataURL(file);
    }

    // Camera handling
    async startCamera() {
        try {
            // Enhanced camera constraints for mobile devices
            const constraints = {
                video: {
                    facingMode: 'environment', // Use back camera on mobile
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    aspectRatio: { ideal: 16/9 }
                }
            };
            
            // Fallback for devices that don't support environment camera
            try {
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (envError) {
                console.log('Environment camera not available, trying default camera');
                const fallbackConstraints = {
                    video: {
                        width: { ideal: 1280, max: 1920 },
                        height: { ideal: 720, max: 1080 }
                    }
                };
                this.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
            }
            
            this.video.srcObject = this.stream;
            this.video.style.display = 'block';
            
            this.startCameraBtn.style.display = 'none';
            this.captureBtn.style.display = 'inline-flex';
            this.stopCameraBtn.style.display = 'inline-flex';
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showError('Could not access camera. Please check permissions.');
        }
    }

    capturePhoto() {
        const context = this.canvas.getContext('2d');
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        context.drawImage(this.video, 0, 0);
        
        this.canvas.toBlob((blob) => {
            this.selectedFile = blob;
            this.previewImg.src = this.canvas.toDataURL();
            this.imagePreview.style.display = 'block';
            this.imagePreview.classList.add('fade-in');
            this.hideResults();
        }, 'image/png');
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.style.display = 'none';
        this.startCameraBtn.style.display = 'inline-flex';
        this.captureBtn.style.display = 'none';
        this.stopCameraBtn.style.display = 'none';
    }

    // Prediction
    async makePrediction() {
        console.log('Making prediction...');
        console.log('Current method:', this.currentMethod);
        console.log('Selected file:', this.selectedFile);
        
        if (!this.selectedFile) {
            this.showError('Please select an image first');
            return;
        }

        this.showLoading(true);
        
        try {
            let response;
            
            if (this.currentMethod === 'camera') {
                console.log('Using camera prediction endpoint');
                // Send base64 image for camera
                const imageData = this.canvas.toDataURL();
                response = await fetch('/predict_camera', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                });
            } else {
                console.log('Using upload prediction endpoint');
                // Send file for upload
                const formData = new FormData();
                formData.append('file', this.selectedFile);
                
                console.log('FormData created with file:', this.selectedFile.name);
                
                response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
            }

            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Prediction result:', result);
            
            if (result.error) {
                throw new Error(result.error);
            }

            this.displayResults(result);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError(`Prediction failed: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        // Store current prediction data for PDF generation
        this.currentPredictionData = result;
        
        // Display the analyzed image if available
        if (result.image_url) {
            this.analyzedImage.src = result.image_url;
            this.analyzedImageSection.style.display = 'block';
        }
        
        // Update main prediction
        this.diseaseName.textContent = result.predicted_class;
        this.diseaseDescription.textContent = result.description;
        this.confidenceValue.textContent = `${result.confidence}%`;
        this.timestamp.textContent = `Analysis completed: ${result.timestamp}`;
        
        // Update confidence badge color
        this.updateConfidenceBadge(result.confidence);
        
        // Update disease icon
        this.updateDiseaseIcon(result.predicted_class);
        
        // Display probabilities
        this.displayProbabilities(result.all_predictions);
        
        // Display recommendations
        this.displayRecommendations(result.recommendations);
        
        // Show results
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('fade-in');
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    updateConfidenceBadge(confidence) {
        if (confidence >= 90) {
            this.confidenceBadge.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
        } else if (confidence >= 70) {
            this.confidenceBadge.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
        } else {
            this.confidenceBadge.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        }
    }

    updateDiseaseIcon(diseaseName) {
        const iconMap = {
            'Early Blight': { icon: 'fas fa-exclamation-triangle', color: '#f59e0b' },
            'Late Blight': { icon: 'fas fa-skull-crossbones', color: '#ef4444' },
            'Healthy': { icon: 'fas fa-check-circle', color: '#22c55e' }
        };
        
        const iconInfo = iconMap[diseaseName] || { icon: 'fas fa-leaf', color: '#667eea' };
        this.diseaseIcon.innerHTML = `<i class="${iconInfo.icon}"></i>`;
        this.diseaseIcon.style.color = iconInfo.color;
    }

    displayProbabilities(allPredictions) {
        this.probabilities.innerHTML = '';
        
        Object.entries(allPredictions).forEach(([disease, data]) => {
            const probability = data.probability;
            const item = document.createElement('div');
            item.className = 'probability-item';
            
            const color = this.getProbabilityColor(probability);
            
            item.innerHTML = `
                <div class="probability-label">${disease}</div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${probability}%; background: ${color};"></div>
                </div>
                <div class="probability-value">${probability}%</div>
            `;
            
            this.probabilities.appendChild(item);
        });
    }

    getProbabilityColor(probability) {
        if (probability >= 70) return 'linear-gradient(90deg, #22c55e, #16a34a)';
        if (probability >= 40) return 'linear-gradient(90deg, #f59e0b, #d97706)';
        return 'linear-gradient(90deg, #ef4444, #dc2626)';
    }

    displayRecommendations(recommendations) {
        this.recommendationList.innerHTML = '';
        
        recommendations.forEach((rec, index) => {
            const item = document.createElement('div');
            item.className = 'recommendation-item';
            item.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <span>${rec}</span>
            `;
            this.recommendationList.appendChild(item);
        });
    }

    // Utility methods
    clearSelection() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.imagePreview.style.display = 'none';
        this.hideResults();
    }

    newAnalysis() {
        this.clearSelection();
        this.stopCamera();
        this.switchMethod('upload');
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
        this.analyzedImageSection.style.display = 'none';
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    showError(message) {
        alert(`Error: ${message}`);
    }

    async downloadReport() {
        try {
            // Show loading state
            this.downloadResultBtn.disabled = true;
            
            if (this.folderSelectionSupported) {
                this.downloadResultBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Preparing folder selection...';
            } else {
                this.downloadResultBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
            }
            
            // Gather all report data
            const reportData = {
                predicted_class: this.diseaseName.textContent,
                confidence: parseFloat(this.confidenceValue.textContent.replace('%', '')),
                description: this.diseaseDescription.textContent,
                timestamp: this.timestamp.textContent,
                image_url: this.analyzedImage.src || null,
                all_predictions: this.currentPredictionData || {},
                recommendations: this.getCurrentRecommendations()
            };
            
            // Try to generate PDF via backend
            const response = await fetch('/generate-pdf-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reportData)
            });
            
            if (response.ok) {
                // Backend PDF generation successful
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                
                // Create download link with File System Access API for folder selection
                if (this.folderSelectionSupported) {
                    this.downloadResultBtn.innerHTML = '<i class="fas fa-folder-open"></i> Choose save location...';
                    
                    try {
                        // Modern browsers with File System Access API
                        const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '');
                        const diseaseName = reportData.predicted_class.replace(/\s+/g, '_');
                        const filename = `potato_disease_report_${diseaseName}_${timestamp}.pdf`;
                        
                        // Show folder picker dialog
                        const fileHandle = await window.showSaveFilePicker({
                            suggestedName: filename,
                            types: [
                                {
                                    description: 'PDF Reports',
                                    accept: {
                                        'application/pdf': ['.pdf'],
                                    },
                                },
                            ],
                            excludeAcceptAllOption: true,
                            startIn: 'documents' // Suggest Documents folder
                        });
                        
                        this.downloadResultBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving to selected folder...';
                        
                        const writable = await fileHandle.createWritable();
                        await writable.write(blob);
                        await writable.close();
                        
                        this.showSuccessMessage('📁 PDF report saved to your chosen folder successfully!');
                    } catch (err) {
                        if (err.name === 'AbortError') {
                            // User cancelled folder selection
                            this.showInfoMessage('📁 Folder selection cancelled. Try again to choose a save location.');
                        } else {
                            console.error('Folder save error:', err);
                            // Fallback to regular download
                            this.fallbackDownload(url, blob, reportData);
                            this.showWarningMessage('📁 Folder selection failed. Downloaded to default location instead.');
                        }
                    }
                } else {
                    // Fallback for browsers without File System Access API
                    this.fallbackDownload(url, blob, reportData);
                }
                
                window.URL.revokeObjectURL(url);
            } else {
                // Backend failed, check if it's a server-side issue or ReportLab missing
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData = { error: 'Unknown server error' };
                }
                
                console.warn('Backend PDF generation failed:', errorData);
                
                if (errorData.fallback === 'client' || response.status === 503) {
                    // Server suggests client-side fallback
                    console.log('Using client-side PDF generation fallback');
                    await this.generateClientSidePDF(reportData);
                } else {
                    // Other server errors
                    throw new Error(errorData.message || errorData.error || 'Server PDF generation failed');
                }
            }
            
        } catch (error) {
            console.error('Error downloading report:', error);
            // Final fallback to text report
            this.generateTextReport();
            this.showErrorMessage('PDF generation failed, downloaded as text file instead.');
        } finally {
            // Reset button state
            this.downloadResultBtn.disabled = false;
            
            if (this.folderSelectionSupported) {
                this.downloadResultBtn.innerHTML = '<i class="fas fa-folder-open"></i> Choose Folder & Download PDF';
            } else {
                this.downloadResultBtn.innerHTML = '<i class="fas fa-download"></i> Download PDF Report';
            }
        }
    }
    
    fallbackDownload(url, blob, reportData) {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '');
        const diseaseName = reportData.predicted_class.replace(/\s+/g, '_');
        const filename = `potato_disease_report_${diseaseName}_${timestamp}.pdf`;
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        this.showSuccessMessage('PDF report downloaded to default folder!');
    }
    
    async generateClientSidePDF(reportData) {
        // Client-side PDF generation using jsPDF
        try {
            if (typeof jsPDF === 'undefined') {
                throw new Error('jsPDF library not loaded');
            }
            
            console.log('📄 Generating client-side PDF...');
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            // Add content to PDF
            doc.setFontSize(20);
            doc.text('🥔 POTATO DISEASE DETECTION REPORT', 20, 30);
            
            doc.setFontSize(12);
            doc.text(`Report Generated: ${reportData.timestamp}`, 20, 50);
            doc.text(`Analysis Method: Deep Learning AI Classification`, 20, 60);
            doc.text(`Model Version: TensorFlow/Keras CNN v1.0`, 20, 70);
            
            // Main diagnosis
            doc.setFontSize(16);
            doc.text('🎯 DIAGNOSIS RESULTS', 20, 90);
            
            doc.setFontSize(12);
            doc.text(`Predicted Disease: ${reportData.predicted_class}`, 20, 105);
            doc.text(`Confidence: ${reportData.confidence}%`, 20, 115);
            
            // Risk assessment
            let riskLevel = 'Unknown';
            if (reportData.confidence >= 80) riskLevel = 'High Confidence';
            else if (reportData.confidence >= 60) riskLevel = 'Medium Confidence';
            else riskLevel = 'Low Confidence - Manual Verification Recommended';
            
            doc.text(`Risk Assessment: ${riskLevel}`, 20, 125);
            
            // Description
            doc.setFontSize(16);
            doc.text('📋 DESCRIPTION', 20, 145);
            
            doc.setFontSize(10);
            const splitDescription = doc.splitTextToSize(reportData.description, 170);
            doc.text(splitDescription, 20, 160);
            
            let yPos = 160 + (splitDescription.length * 5) + 15;
            
            // Probability breakdown
            doc.setFontSize(16);
            doc.text('📊 PROBABILITY BREAKDOWN', 20, yPos);
            yPos += 15;
            
            doc.setFontSize(10);
            if (reportData.all_predictions) {
                for (const [disease, info] of Object.entries(reportData.all_predictions)) {
                    doc.text(`• ${disease}: ${info.probability}%`, 20, yPos);
                    yPos += 10;
                }
            }
            
            yPos += 10;
            
            // Recommendations
            doc.setFontSize(16);
            doc.text('💡 TREATMENT RECOMMENDATIONS', 20, yPos);
            yPos += 15;
            
            doc.setFontSize(10);
            reportData.recommendations.forEach((rec, index) => {
                const recText = `${index + 1}. ${rec}`;
                const splitRec = doc.splitTextToSize(recText, 170);
                doc.text(splitRec, 20, yPos);
                yPos += splitRec.length * 5 + 3;
                
                // Add new page if needed
                if (yPos > 270) {
                    doc.addPage();
                    yPos = 20;
                }
            });
            
            // Footer
            yPos = Math.max(yPos + 20, 250);
            doc.setFontSize(8);
            doc.text('Generated by Potato Disease Detection System', 20, yPos);
            doc.text('Powered by Flask & TensorFlow | Lucky Sharma', 20, yPos + 8);
            doc.text('© 2025 All Rights Reserved', 20, yPos + 16);
            
            // Save PDF
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '');
            const diseaseName = reportData.predicted_class.replace(/\s+/g, '_');
            const filename = `potato_disease_report_${diseaseName}_${timestamp}.pdf`;
            
            // Try to use File System Access API for folder selection
            if ('showSaveFilePicker' in window) {
                try {
                    const fileHandle = await window.showSaveFilePicker({
                        suggestedName: filename,
                        types: [
                            {
                                description: 'PDF files',
                                accept: {
                                    'application/pdf': ['.pdf'],
                                },
                            },
                        ],
                    });
                    
                    const writable = await fileHandle.createWritable();
                    const pdfBlob = doc.output('blob');
                    await writable.write(pdfBlob);
                    await writable.close();
                    
                    this.showSuccessMessage('PDF report saved successfully using client-side generation!');
                } catch (err) {
                    if (err.name !== 'AbortError') {
                        // Fallback to regular download
                        doc.save(filename);
                        this.showSuccessMessage('PDF report generated successfully!');
                    }
                }
            } else {
                // Regular download for older browsers
                doc.save(filename);
                this.showSuccessMessage('PDF report generated successfully!');
            }
            
        } catch (error) {
            console.error('Client-side PDF generation failed:', error);
            this.showErrorMessage('PDF generation failed. Falling back to text report.');
            this.generateTextReport();
        }
    }
    
    getCurrentRecommendations() {
        const recommendations = [];
        const recItems = this.recommendationList.querySelectorAll('.recommendation-item span');
        recItems.forEach(item => {
            recommendations.push(item.textContent);
        });
        return recommendations;
    }
    
    generateTextReport() {
        // Fallback text report generation (original functionality)
        const diseaseName = this.diseaseName.textContent;
        const confidence = this.confidenceValue.textContent;
        const description = this.diseaseDescription.textContent;
        const timestamp = this.timestamp.textContent;
        
        let report = `POTATO DISEASE DETECTION REPORT\n`;
        report += `=====================================\n\n`;
        report += `${timestamp}\n\n`;
        report += `DIAGNOSIS: ${diseaseName}\n`;
        report += `CONFIDENCE: ${confidence}\n\n`;
        report += `DESCRIPTION:\n${description}\n\n`;
        report += `RECOMMENDATIONS:\n`;
        
        const recommendations = this.recommendationList.querySelectorAll('.recommendation-item span');
        recommendations.forEach((rec, index) => {
            report += `${index + 1}. ${rec.textContent}\n`;
        });
        
        report += `\n=====================================\n`;
        report += `Generated by Potato Disease Detection System\n`;
        report += `Powered by Flask & TensorFlow\n`;
        
        // Download as text file
        const blob = new Blob([report], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `potato_disease_report_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    showSuccessMessage(message) {
        this.showMessage(message, 'success');
    }

    showInfoMessage(message) {
        this.showMessage(message, 'info');
    }

    showWarningMessage(message) {
        this.showMessage(message, 'warning');
    }

    showErrorMessage(message) {
        this.showMessage(message, 'error');
    }

    showMessage(message, type = 'info') {
        // Create or update message container
        let messageContainer = document.getElementById('message-container');
        if (!messageContainer) {
            messageContainer = document.createElement('div');
            messageContainer.id = 'message-container';
            messageContainer.style.position = 'fixed';
            messageContainer.style.top = '20px';
            messageContainer.style.right = '20px';
            messageContainer.style.zIndex = '10000';
            messageContainer.style.maxWidth = '400px';
            document.body.appendChild(messageContainer);
        }

        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${type}`;
        messageEl.innerHTML = `
            <div style="
                background: ${this.getMessageColor(type)};
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 14px;
                animation: slideInRight 0.3s ease-out;
            ">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; color: white; font-size: 18px; cursor: pointer; padding: 0; margin-left: 10px;">×</button>
            </div>
        `;

        // Add CSS animation if not already added
        if (!document.getElementById('message-styles')) {
            const style = document.createElement('style');
            style.id = 'message-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }

        messageContainer.appendChild(messageEl);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageEl.parentElement) {
                messageEl.remove();
            }
        }, 5000);
    }

    getMessageColor(type) {
        const colors = {
            success: '#10b981', // green
            info: '#3b82f6',    // blue
            warning: '#f59e0b', // amber
            error: '#ef4444'    // red
        };
        return colors[type] || colors.info;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PotatoDiseaseDetector();
});

// Add some utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Mobile device detection and utilities
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function isIOSDevice() {
    return /iPad|iPhone|iPod/.test(navigator.userAgent);
}

function getOptimalImageSize() {
    const isMobile = isMobileDevice();
    if (isMobile) {
        return {
            maxWidth: window.innerWidth - 40,
            maxHeight: Math.min(window.innerHeight * 0.4, 300)
        };
    }
    return {
        maxWidth: 400,
        maxHeight: 400
    };
}

// Prevent double-tap zoom on mobile
function preventDoubleTab() {
    let lastTouchEnd = 0;
    document.addEventListener('touchend', function (event) {
        const now = (new Date()).getTime();
        if (now - lastTouchEnd <= 300) {
            event.preventDefault();
        }
        lastTouchEnd = now;
    }, false);
}

// Initialize mobile optimizations
if (isMobileDevice()) {
    preventDoubleTab();
    
    // Add mobile class to body for CSS targeting
    document.body.classList.add('mobile-device');
    
    if (isIOSDevice()) {
        document.body.classList.add('ios-device');
    }
    
    // Adjust viewport height for mobile browsers
    function setVH() {
        let vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }
    
    setVH();
    window.addEventListener('resize', setVH);
    window.addEventListener('orientationchange', () => {
        setTimeout(setVH, 100);
    });
}

// Check browser compatibility
function checkBrowserSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('Camera functionality not supported in this browser');
        const cameraCard = document.getElementById('cameraCard');
        if (cameraCard) {
            cameraCard.style.opacity = '0.5';
            cameraCard.style.cursor = 'not-allowed';
            
            // Add tooltip for unsupported browsers
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = 'Camera not supported in this browser';
            cameraCard.appendChild(tooltip);
        }
    }
    
    // Check for file upload support
    if (!window.File || !window.FileReader || !window.FileList || !window.Blob) {
        console.warn('File upload not supported in this browser');
        const uploadCard = document.getElementById('uploadCard');
        if (uploadCard) {
            uploadCard.style.opacity = '0.7';
        }
    }
}

// Run compatibility check
checkBrowserSupport();
