document.addEventListener('DOMContentLoaded', function () {

    /* ===============================
       Body Type Dropdown - Dynamic Filtering by Gender
    =============================== */
    const bodyTypeSelect = document.getElementById('bodyType');
    
    function updateBodyTypeOptions() {
        const selectedGender = document.querySelector('input[name="gender"]:checked').value;
        const options = bodyTypeSelect.querySelectorAll('option[data-gender]');
        
        // Reset selection when gender changes
        bodyTypeSelect.value = '';
        
        // Show/hide options based on selected gender
        options.forEach(option => {
            if (option.dataset.gender === selectedGender) {
                option.style.display = 'block';
            } else {
                option.style.display = 'none';
            }
        });
    }

    /* ===============================
       Gender selection
    =============================== */
    document.querySelectorAll('.gender-option').forEach(function (option) {
        option.addEventListener('click', function () {
            document.querySelectorAll('.gender-option').forEach(function (o) {
                o.classList.remove('selected');
            });
            this.classList.add('selected');
            this.querySelector('input').checked = true;
            
            // Update body type dropdown when gender changes
            updateBodyTypeOptions();
        });
    });
    
    // Initialize body type options on page load
    updateBodyTypeOptions();

    /* ===============================
       Upload setup
    =============================== */
    function setupUpload(boxId, inputId, previewId, nameId) {
        var box = document.getElementById(boxId);
        var input = document.getElementById(inputId);
        var preview = document.getElementById(previewId);
        var name = document.getElementById(nameId);

        if (!box || !input) return;

        box.addEventListener('click', function () {
            input.click();
        });

        input.addEventListener('change', function () {
            if (this.files && this.files[0]) {
                var file = this.files[0];
                if (name) name.textContent = file.name;
                box.classList.add('has-file');

                var reader = new FileReader();
                reader.onload = function (e) {
                    if (preview) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }

    setupUpload('frontUpload', 'frontImage', 'frontPreview', 'frontFileName');
    setupUpload('sideUpload', 'sideImage', 'sidePreview', 'sideFileName');

    /* ===============================
       Helpers
    =============================== */
    function formatBodyType(bodyType) {
        if (!bodyType) return '-';
        return bodyType.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function getBMICategoryClass(category) {
        const classes = {
            underweight: 'bmi-underweight',
            normal: 'bmi-normal',
            overweight: 'bmi-overweight',
            obese: 'bmi-obese'
        };
        return classes[category] || '';
    }

    function safeUpdate(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    /* ===============================
       Enhanced Error Display Function
    =============================== */
    function displayError(message) {
        const errorDiv = document.getElementById('error');
        if (!errorDiv) return;

        // Format the error message properly
        let formattedMessage = message
            // Convert double newlines to paragraph breaks
            .replace(/\n\n/g, '<br><br>')
            // Convert single newlines to line breaks
            .replace(/\n/g, '<br>')
            // Style emoji icons
            .replace(/❌/g, '<span style="font-size: 1.2em;">❌</span>')
            .replace(/💡/g, '<span style="font-size: 1.2em;">💡</span>')
            .replace(/⚠️/g, '<span style="font-size: 1.2em;">⚠️</span>')
            .replace(/✓/g, '<span style="color: #10b981;">✓</span>')
            // Make bullet points more visible
            .replace(/• /g, '<span style="color: #dc2626; font-weight: bold;">• </span>');

        // Use innerHTML to render HTML tags
        errorDiv.innerHTML = formattedMessage;
        errorDiv.classList.add('active');

        // Scroll to error message smoothly
        setTimeout(() => {
            errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    }

    /* ===============================
       Form submission
    =============================== */
    const form = document.getElementById('measurementForm');

    if (!form) return;

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const formData = new FormData(this);
        const btn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const error = document.getElementById('error');

        // Hide previous results and errors
        if (results) results.classList.remove('active');
        if (error) error.classList.remove('active');

        // Disable button and show loading
        if (btn) btn.disabled = true;
        if (loading) loading.classList.add('active');

        /* Initialize 3D spinner */
        setTimeout(() => {
            if (typeof init3DSpinner === 'function') {
                try {
                    init3DSpinner('spinner3d');
                } catch (err) {
                    console.error('Spinner error:', err);
                }
            }
        }, 50);

        try {
            const res = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!data.success) {
                // Use the enhanced error display function
                throw new Error(data.error || 'Processing failed');
            }

            const m = data.measurements;

            /* ===============================
               Metadata (BMI / Body Type / Size)
            =============================== */
            if (m.metadata) {
                const meta = m.metadata;

                // BMI
                safeUpdate('bmiValue', meta.bmi);

                const bmiCategoryEl = document.getElementById('bmiCategory');
                if (bmiCategoryEl && meta.bmi_category) {
                    const text = meta.bmi_category.charAt(0).toUpperCase() + meta.bmi_category.slice(1);
                    bmiCategoryEl.textContent = text;
                    bmiCategoryEl.className = 'result-category ' + getBMICategoryClass(meta.bmi_category);
                }

                // Body Type (calculated) - FIXED ID
                safeUpdate('bodyTypeResult', formatBodyType(meta.body_type));

                // Recommended Size
                safeUpdate('recommendedSize', meta.recommended_size || '-');
                
                // Log user's selected body type (for debugging)
                if (meta.body_type_input) {
                    console.log('User selected body type:', meta.body_type_input);
                }
            }

            /* ===============================
               Measurements
            =============================== */

            // Neck Circumference
            safeUpdate('neckCircCm', m.neck?.circumference?.cm);
            safeUpdate('neckCircIn', m.neck?.circumference?.inches);

            // Chest Circumference
            safeUpdate('chestCircCm', m.chest?.circumference?.cm);
            safeUpdate('chestCircIn', m.chest?.circumference?.inches);

            // Chest Circumference
            safeUpdate('chestCircCm', m.chest?.circumference?.cm);
            safeUpdate('chestCircIn', m.chest?.circumference?.inches);

            // Upper Chest Circumference (Females Only)
            const upperChestCard = document.getElementById('upperChestCard');
            if (m.upper_chest && upperChestCard) {
                safeUpdate('upperChestCircCm', m.upper_chest?.circumference?.cm);
                safeUpdate('upperChestCircIn', m.upper_chest?.circumference?.inches);
                upperChestCard.style.display = 'block';
                upperChestCard.classList.add('show');
            } else if (upperChestCard) {
                upperChestCard.style.display = 'none';
            }

            // Lower Chest Circumference (Females Only)
            const lowerChestCard = document.getElementById('lowerChestCard');
            if (m.lower_chest && lowerChestCard) {
                safeUpdate('lowerChestCircCm', m.lower_chest?.circumference?.cm);
                safeUpdate('lowerChestCircIn', m.lower_chest?.circumference?.inches);
                lowerChestCard.style.display = 'block';
                lowerChestCard.classList.add('show');
            } else if (lowerChestCard) {
                lowerChestCard.style.display = 'none';
            }

            // Waist Circumference
            safeUpdate('waistCircCm', m.waist?.circumference?.cm);
            safeUpdate('waistCircIn', m.waist?.circumference?.inches);

            // Waist Circumference
            safeUpdate('waistCircCm', m.waist?.circumference?.cm);
            safeUpdate('waistCircIn', m.waist?.circumference?.inches);

            // Hip Circumference
            safeUpdate('hipCircCm', m.hip?.circumference?.cm);
            safeUpdate('hipCircIn', m.hip?.circumference?.inches);

            // Shoulder Width
            safeUpdate('shoulderWidthCm', m.shoulder?.width?.cm);
            safeUpdate('shoulderWidthIn', m.shoulder?.width?.inches);

            // Arm Lengths
            safeUpdate('armHandCm', m.arm?.hand_to_elbow?.cm);
            safeUpdate('armHandIn', m.arm?.hand_to_elbow?.inches);

            // Shoulder width 
            safeUpdate('armShoulderCm', m.arm?.shoulder_to_elbow?.cm);
            safeUpdate('armShoulderIn', m.arm?.shoulder_to_elbow?.inches);

            // Total arm length
            safeUpdate('armTotalCm', m.arm?.total_length?.cm);
            safeUpdate('armTotalIn', m.arm?.total_length?.inches);

            // Armhole circumference
            safeUpdate('armholeCircCm', m.armhole?.circumference?.cm);
            safeUpdate('armholeCircIn', m.armhole?.circumference?.inches);

            // Thigh circumference
            safeUpdate('upperThighCircCm', m.upper_thigh?.circumference?.cm);
            safeUpdate('upperThighCircIn', m.upper_thigh?.circumference?.inches);

            // Knee circumference
            safeUpdate('kneeCircCm', m.knee?.circumference?.cm);
            safeUpdate('kneeCircIn', m.knee?.circumference?.inches);

            // Body Length
            safeUpdate('bodyLengthCm', m.body_length?.length?.cm);
            safeUpdate('bodyLengthIn', m.body_length?.length?.inches);

            // Show results with smooth scroll
            if (results) {
                results.classList.add('active');
                setTimeout(() => {
                    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            }

        } catch (err) {
            console.error('Error:', err);
            
            // Use enhanced error display function
            displayError(err.message || 'Network error. Please try again.');
        }

        // Re-enable form
        if (loading) loading.classList.remove('active');
        if (btn) btn.disabled = false;
    });
});