document.addEventListener('DOMContentLoaded', function () {

    /* ===============================
       Multi-Step Form State
    =============================== */
    let currentStep = 1;
    const totalSteps = 6;
    
    const startContainer = document.getElementById('startContainer');
    const formContainer = document.getElementById('formContainer');
    const startBtn = document.getElementById('startMeasurementBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    const progressFill = document.getElementById('progressFill');
    const currentStepSpan = document.getElementById('currentStep');

    // Store calculated values
    let calculatedBMI = null;
    let userAge = null;

    /* ===============================
       Helper Functions
    =============================== */
    function getAgeGroup(age) {
        if (age < 18) return 'teen';
        if (age >= 18 && age < 45) return 'adult';
        if (age >= 45 && age < 65) return 'middle_age';
        return 'senior';
    }

    /* ===============================
       BMI Calculation
    =============================== */
    function calculateBMI() {
        const height = parseFloat(document.getElementById('heightInput').value);
        const heightUnit = document.getElementById('heightUnit').value;
        const weight = parseFloat(document.getElementById('weightInput').value);
        const weightUnit = document.getElementById('weightUnit').value;

        if (!height || !weight) return null;

        // Convert to metric (cm and kg)
        let heightInCm = height;
        let weightInKg = weight;

        if (heightUnit === 'm') {
            heightInCm = height * 100;
        }
        if (weightUnit === 'lbs') {
            weightInKg = weight * 0.453592;
        }

        // Calculate BMI
        const heightInMeters = heightInCm / 100;
        const bmi = weightInKg / (heightInMeters * heightInMeters);
        
        return parseFloat(bmi.toFixed(1));
    }

    function getBMICategoryFromValue(bmi) {
        if (bmi < 18.5) return 'underweight';
        if (bmi < 25) return 'normal';
        if (bmi < 30) return 'overweight';
        return 'obese';
    }

    /* ===============================
       Step Visibility Logic
    =============================== */
    function shouldShowStep(stepNumber) {
        // Step 1 (Basic Info) - Always visible
        if (stepNumber === 1) return true;

        // Step 2 (Fat Distribution, Body Type) - BMI ≥ normal (18.5)
        if (stepNumber === 2) {
            return calculatedBMI !== null && calculatedBMI >= 18.5;
        }

        // Step 3 (Activity Level, Muscle Level) - BMI ≥ normal
        if (stepNumber === 3) {
            return calculatedBMI !== null && calculatedBMI >= 18.5;
        }

        // Step 4 (Fit & Goal) - Always visible
        if (stepNumber === 4) return true;

        // Step 5 (Shoulder Type) - age ≥ 18 AND BMI ≠ underweight
        if (stepNumber === 5) {
            return userAge !== null && 
                   userAge >= 18 && 
                   calculatedBMI !== null && 
                   calculatedBMI >= 18.5;
        }

        // Step 6 (Image Upload) - Always visible (last)
        if (stepNumber === 6) return true;

        return true;
    }

    function updateStepVisibility() {
        // Recalculate BMI and get age
        calculatedBMI = calculateBMI();
        userAge = parseInt(document.getElementById('age').value) || null;

        // Update visibility for each step
        for (let i = 1; i <= 6; i++) {
            const stepEl = document.querySelector(`.form-step[data-step="${i}"]`);
            if (stepEl) {
                const shouldShow = shouldShowStep(i);
                if (shouldShow) {
                    stepEl.removeAttribute('data-hidden');
                    // Restore required attributes
                    const inputs = stepEl.querySelectorAll('[data-required="true"]');
                    inputs.forEach(input => {
                        input.setAttribute('required', 'required');
                    });
                } else {
                    stepEl.setAttribute('data-hidden', 'true');
                    // Remove required validation from hidden steps
                    const inputs = stepEl.querySelectorAll('input[required], select[required]');
                    inputs.forEach(input => {
                        input.setAttribute('data-required', 'true');
                        input.removeAttribute('required');
                    });
                }
            }
        }
    }

    function getVisibleSteps() {
        const visible = [];
        for (let i = 1; i <= 6; i++) {
            if (shouldShowStep(i)) {
                visible.push(i);
            }
        }
        return visible;
    }

    function getNextVisibleStep(fromStep) {
        for (let i = fromStep + 1; i <= 6; i++) {
            if (shouldShowStep(i)) return i;
        }
        return fromStep;
    }

    function getPreviousVisibleStep(fromStep) {
        for (let i = fromStep - 1; i >= 1; i--) {
            if (shouldShowStep(i)) return i;
        }
        return fromStep;
    }

    /* ===============================
       Watch for BMI/Age Changes on Step 1
    =============================== */
    const heightInput = document.getElementById('heightInput');
    const heightUnit = document.getElementById('heightUnit');
    const weightInput = document.getElementById('weightInput');
    const weightUnit = document.getElementById('weightUnit');
    const ageInput = document.getElementById('age');

    function handleStep1Change() {
        updateStepVisibility();
        
        // Get age group for backend
        const age = parseInt(document.getElementById('age').value) || null;
        const ageGroup = age ? getAgeGroup(age) : null;
        
        // Store in a hidden field for backend
        let ageGroupInput = document.getElementById('age_group_hidden');
        if (!ageGroupInput) {
            ageGroupInput = document.createElement('input');
            ageGroupInput.type = 'hidden';
            ageGroupInput.id = 'age_group_hidden';
            ageGroupInput.name = 'age_group';
            document.getElementById('measurementForm').appendChild(ageGroupInput);
        }
        ageGroupInput.value = ageGroup || '';
        
        console.log('BMI:', calculatedBMI, 'Age:', userAge, 'Age Group:', ageGroup, 'Visible Steps:', getVisibleSteps());
    }

    if (heightInput) heightInput.addEventListener('input', handleStep1Change);
    if (heightUnit) heightUnit.addEventListener('change', handleStep1Change);
    if (weightInput) weightInput.addEventListener('input', handleStep1Change);
    if (weightUnit) weightUnit.addEventListener('change', handleStep1Change);
    if (ageInput) ageInput.addEventListener('input', handleStep1Change);

    /* ===============================
       Start Measurement Button
    =============================== */
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            startContainer.style.display = 'none';
            formContainer.style.display = 'block';
            updateStepVisibility(); // Initialize visibility
            updateStepDisplay();
        });
    }

    /* ===============================
       Step Navigation Functions
    =============================== */
    function updateStepDisplay() {
        // Update visibility first
        updateStepVisibility();
        
        // Hide all steps
        document.querySelectorAll('.form-step').forEach(step => {
            step.classList.remove('active');
        });
        
        // Show current step
        const currentStepEl = document.querySelector(`.form-step[data-step="${currentStep}"]`);
        if (currentStepEl) {
            currentStepEl.classList.add('active');
        }
        
        // Calculate visible steps for progress
        const visibleSteps = getVisibleSteps();
        const currentPosition = visibleSteps.indexOf(currentStep) + 1;
        const totalVisible = visibleSteps.length;
        
        // Update progress bar
        const progress = (currentPosition / totalVisible) * 100;
        if (progressFill) progressFill.style.width = progress + '%';
        if (currentStepSpan) currentStepSpan.textContent = currentPosition + ' of ' + totalVisible;
        
        // Update buttons
        const isFirstVisible = currentStep === visibleSteps[0];
        const isLastVisible = currentStep === visibleSteps[visibleSteps.length - 1];
        
        if (prevBtn) prevBtn.style.display = isFirstVisible ? 'none' : 'block';
        if (nextBtn) nextBtn.style.display = isLastVisible ? 'none' : 'block';
        if (submitBtn) submitBtn.style.display = isLastVisible ? 'block' : 'none';
        
        // Scroll to top of form
        if (formContainer) {
            formContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function validateCurrentStep() {
        const currentStepEl = document.querySelector(`.form-step[data-step="${currentStep}"]`);
        if (!currentStepEl) return true;
        
        // Skip validation if step is hidden
        if (currentStepEl.hasAttribute('data-hidden')) return true;
        
        const inputs = currentStepEl.querySelectorAll('input[required], select[required]');
        let isValid = true;
        let missingFields = [];
        
        inputs.forEach(input => {
            if (input.type === 'radio') {
                const radioGroup = currentStepEl.querySelectorAll(`input[name="${input.name}"]`);
                const isChecked = Array.from(radioGroup).some(radio => radio.checked);
                if (!isChecked) {
                    isValid = false;
                    missingFields.push(input.name);
                }
            } else if (input.type === 'file') {
                if (!input.files || input.files.length === 0) {
                    isValid = false;
                    missingFields.push(input.name);
                }
            } else if (!input.value.trim()) {
                isValid = false;
                missingFields.push(input.name || input.id);
            }
        });
        
        if (!isValid) {
            alert('Please fill in all required fields before proceeding.');
            console.log('Missing fields:', missingFields);
        }
        
        return isValid;
    }

    /* ===============================
       Next/Previous Button Handlers
    =============================== */
    // Next Button
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            if (validateCurrentStep()) {
                const nextStep = getNextVisibleStep(currentStep);
                if (nextStep !== currentStep) {
                    currentStep = nextStep;
                    updateStepDisplay();
                }
            }
        });
    }

    // Previous Button
    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            const prevStep = getPreviousVisibleStep(currentStep);
            if (prevStep !== currentStep) {
                currentStep = prevStep;
                updateStepDisplay();
            }
        });
    }

    /* ===============================
       Body Type Dropdown - Dynamic Filtering by Gender
    =============================== */
    const bodyTypeSelect = document.getElementById('bodyType');
    
    function updateBodyTypeOptions() {
        const genderInput = document.querySelector('input[name="gender"]:checked');
        if (!genderInput) return;
        
        const selectedGender = genderInput.value;
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
    if (bodyTypeSelect) {
        updateBodyTypeOptions();
    }

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

        // Final validation for multi-step
        if (!validateCurrentStep()) return;

        const formData = new FormData(this);
        
        // ADD DEFAULT VALUES FOR CONDITIONALLY HIDDEN FIELDS
        // If BMI < 18.5, these fields won't exist, so provide defaults
        if (calculatedBMI !== null && calculatedBMI < 18.5) {
            if (!formData.has('fat_distribution')) {
                formData.set('fat_distribution', 'even'); // default
            }
            if (!formData.has('body_type')) {
                formData.set('body_type', 'slim'); // default for underweight
            }
            if (!formData.has('activity_level')) {
                formData.set('activity_level', 'light'); // default
            }
            if (!formData.has('muscle_level')) {
                formData.set('muscle_level', 'low'); // default
            }
        }
        
        // If age < 18 or BMI < 18.5, shoulder_type won't exist
        if (!formData.has('shoulder_type')) {
            formData.set('shoulder_type', 'average'); // default
        }
        
        // Ensure age_group is set
        const age = parseInt(document.getElementById('age').value);
        formData.set('age_group', getAgeGroup(age));

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

                // Body Type (calculated)
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

            // Hip Circumference
            safeUpdate('hipCircCm', m.hip?.circumference?.cm);
            safeUpdate('hipCircIn', m.hip?.circumference?.inches);

            // Shoulder Width
            safeUpdate('shoulderWidthCm', m.shoulder?.width?.cm);
            safeUpdate('shoulderWidthIn', m.shoulder?.width?.inches);

            // Arm Lengths
            safeUpdate('armHandCm', m.arm?.hand_to_elbow?.cm);
            safeUpdate('armHandIn', m.arm?.hand_to_elbow?.inches);

            safeUpdate('armShoulderCm', m.arm?.shoulder_to_elbow?.cm);
            safeUpdate('armShoulderIn', m.arm?.shoulder_to_elbow?.inches);

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