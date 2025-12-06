// ImageRevive UI JavaScript with Resolution Selection

let selectedFile = null;
let jobId = null;
let inputAspectRatio = 1;

// File upload handling
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const uploadPreview = document.getElementById('uploadPreview');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const previewSection = document.getElementById('previewSection');
const resolutionSection = document.getElementById('resolutionSection');

fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect(e) {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        showImagePreview(selectedFile);
        processBtn.disabled = false;
        uploadSection.classList.add('has-image');
    }
}

function showImagePreview(file) {
    // Display file info
    const fileSize = (file.size / (1024 * 1024)).toFixed(2); // MB
    document.getElementById('imageSize').textContent = fileSize + ' MB';
    document.getElementById('imageType').textContent = file.type || 'Unknown';
    
    // Read and display image
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('uploadedImage');
        img.src = e.target.result;
        
        // Also set for original preview
        document.getElementById('originalPreview').src = e.target.result;
        
        // Get image dimensions and aspect ratio
        img.onload = function() {
            const width = this.naturalWidth;
            const height = this.naturalHeight;
            inputAspectRatio = width / height;
            
            document.getElementById('imageDimensions').textContent = 
                width + ' × ' + height + ' px';
            document.getElementById('imageAspectRatio').textContent = 
                inputAspectRatio.toFixed(2) + ':1';
            document.getElementById('inputDims').textContent = 
                width + ' × ' + height + ' px';
            
            // Update custom resolution placeholders
            updateCustomResolutionPlaceholders();
        };
        
        // Show preview container
        uploadPreview.classList.add('active');
    };
    reader.readAsDataURL(file);
}

function updateCustomResolutionPlaceholders() {
    // Calculate sample custom resolution based on aspect ratio
    const sampleHeight = 6000;
    const sampleWidth = Math.round(sampleHeight * inputAspectRatio);
    
    document.getElementById('customWidth').placeholder = `e.g., ${sampleWidth}`;
    document.getElementById('customHeight').placeholder = `e.g., ${sampleHeight}`;
}

// Drag and drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect({ target: { files } });
    }
});

// Task selection
document.querySelectorAll('.task-option').forEach(option => {
    option.addEventListener('click', function(e) {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'LABEL') {
            const checkbox = this.querySelector('input[type="checkbox"]');
            checkbox.checked = !checkbox.checked;
            this.classList.toggle('selected', checkbox.checked);
            
            // Show/hide resolution section based on super-resolution selection
            updateResolutionVisibility();
        }
    });
    
    const checkbox = option.querySelector('input[type="checkbox"]');
    checkbox.addEventListener('change', function() {
        option.classList.toggle('selected', this.checked);
        updateResolutionVisibility();
    });
});

function updateResolutionVisibility() {
    const srSelected = document.getElementById('task-sr').checked;
    resolutionSection.classList.toggle('active', srSelected);
}

// Initialize resolution visibility
updateResolutionVisibility();

// Resolution selection
document.querySelectorAll('.resolution-option').forEach(option => {
    option.addEventListener('click', function() {
        // Deselect all
        document.querySelectorAll('.resolution-option').forEach(opt => {
            opt.classList.remove('selected');
        });
        
        // Select this one
        this.classList.add('selected');
        const radio = this.querySelector('input[type="radio"]');
        radio.checked = true;
        
        // Show/hide custom resolution inputs
        const customSection = document.getElementById('customResolution');
        if (radio.value === 'custom') {
            customSection.classList.add('active');
        } else {
            customSection.classList.remove('active');
        }
    });
});

// Process button
processBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    const tasks = [];
    document.querySelectorAll('.task-option input:checked').forEach(cb => {
        const taskMap = {
            'task-denoising': 'denoising',
            'task-sr': 'super_resolution',
            'task-color': 'colorization',
            'task-inpaint': 'inpainting'
        };
        tasks.push(taskMap[cb.id]);
    });

    if (tasks.length === 0) {
        alert('Please select at least one restoration task');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('tasks', tasks.join(','));
    
    // Add resolution settings if super-resolution is selected
    if (tasks.includes('super_resolution')) {
        const selectedResolution = document.querySelector('input[name="resolution"]:checked').value;
        formData.append('resolution', selectedResolution);
        
        if (selectedResolution === 'custom') {
            const customWidth = document.getElementById('customWidth').value;
            const customHeight = document.getElementById('customHeight').value;
            
            if (!customWidth && !customHeight) {
                alert('Please enter at least one custom dimension (width or height)');
                return;
            }
            
            if (customWidth) formData.append('custom_width', customWidth);
            if (customHeight) formData.append('custom_height', customHeight);
        }
    }

    progressSection.classList.add('active');
    processBtn.disabled = true;

    try {
        const response = await fetch('/api/restore', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.error) {
            showError('Error: ' + data.error);
            processBtn.disabled = false;
            return;
        }
        
        jobId = data.job_id;
        checkStatus(jobId);
        
    } catch (error) {
        showError('Error uploading image: ' + error.message);
        processBtn.disabled = false;
    }
});

// Check job status
async function checkStatus(jobId) {
    try {
        const response = await fetch(`/api/status/${jobId}`);
        const data = await response.json();

        updateProgress(data.progress || 0);

        if (data.status === 'completed') {
            showSuccess('Restoration completed!');
            loadResult(jobId, data);
        } else if (data.status === 'failed') {
            showError('Restoration failed: ' + (data.error || 'Unknown error'));
            processBtn.disabled = false;
        } else {
            setTimeout(() => checkStatus(jobId), 1000);
        }
    } catch (error) {
        showError('Error checking status: ' + error.message);
        processBtn.disabled = false;
    }
}

function updateProgress(percent) {
    const fill = document.getElementById('progressFill');
    fill.style.width = percent + '%';
    fill.textContent = percent + '%';
}

async function loadResult(jobId, statusData) {
    const downloadUrl = `/api/download/${jobId}`;
    
    // Load restored image
    const img = document.getElementById('restoredPreview');
    img.src = downloadUrl;
    
    // Get output dimensions when image loads
    img.onload = function() {
        const outputWidth = this.naturalWidth;
        const outputHeight = this.naturalHeight;
        const outputAspect = outputWidth / outputHeight;
        
        document.getElementById('outputDims').textContent = 
            `${outputWidth} × ${outputHeight} px (${outputAspect.toFixed(2)}:1)`;
    };

    // Show preview and download button
    previewSection.classList.add('active');
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = 'inline-block';
    downloadBtn.onclick = () => window.location.href = downloadUrl;

    // Show metrics if available
    if (statusData.quality_score !== undefined) {
        const metricsBox = document.getElementById('metricsBox');
        const metricsContent = document.getElementById('metricsContent');
        metricsBox.style.display = 'block';
        metricsContent.innerHTML = `
            <div class="metric-item">
                <span class="metric-label">Quality Score</span>
                <span class="metric-value">${statusData.quality_score.toFixed(2)}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Processing Time</span>
                <span class="metric-value">${statusData.processing_time || 'N/A'}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Resolution Increase</span>
                <span class="metric-value">${calculateResolutionIncrease()}x</span>
            </div>
        `;
    }

    processBtn.disabled = false;
}

function calculateResolutionIncrease() {
    const inputImg = document.getElementById('uploadedImage');
    const outputImg = document.getElementById('restoredPreview');
    
    if (inputImg.naturalWidth && outputImg.naturalWidth) {
        const inputPixels = inputImg.naturalWidth * inputImg.naturalHeight;
        const outputPixels = outputImg.naturalWidth * outputImg.naturalHeight;
        return (outputPixels / inputPixels).toFixed(1);
    }
    return 'N/A';
}

function showSuccess(message) {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.className = 'status-message success';
    statusDiv.textContent = message;
}

function showError(message) {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.className = 'status-message error';
    statusDiv.textContent = message;
}
