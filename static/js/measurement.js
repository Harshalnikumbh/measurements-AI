document.addEventListener('DOMContentLoaded', function () {
    // Gender selection
    document.querySelectorAll('.gender-option').forEach(function (option) {
        option.addEventListener('click', function () {
            document.querySelectorAll('.gender-option').forEach(function (o) {
                o.classList.remove('selected');
            });
            this.classList.add('selected');
            this.querySelector('input').checked = true;
        });
    });

    // Upload setup
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
                name.textContent = file.name;
                box.classList.add('has-file');

                var reader = new FileReader();
                reader.onload = function (e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    }

    setupUpload('frontUpload', 'frontImage', 'frontPreview', 'frontFileName');
    setupUpload('sideUpload', 'sideImage', 'sidePreview', 'sideFileName');

    // Helper function to format body type display
    function formatBodyType(bodyType) {
        return bodyType.split('_').map(function(word) {
            return word.charAt(0).toUpperCase() + word.slice(1);
        }).join(' ');
    }

    // Helper function to get BMI category color class
    function getBMICategoryClass(category) {
        var classes = {
            'underweight': 'bmi-underweight',
            'normal': 'bmi-normal',
            'overweight': 'bmi-overweight',
            'obese': 'bmi-obese'
        };
        return classes[category] || '';
    }

    // Safe element update - only update if element exists
    function safeUpdate(elementId, value) {
        var element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        } else {
            console.warn('Element not found:', elementId);
        }
    }

    // Form submission
    var form = document.getElementById('measurementForm');
    if (form) {
        form.addEventListener('submit', async function (e) {
            e.preventDefault();

            var formData = new FormData(this);
            var btn = document.getElementById('submitBtn');
            var loading = document.getElementById('loading');
            var results = document.getElementById('results');
            var error = document.getElementById('error');

            // Reset UI
            results.classList.remove('active');
            error.classList.remove('active');
            btn.disabled = true;

            // Show loading
            loading.classList.add('active');
            
            // Initialize 3D spinner
            setTimeout(function() {
                if (typeof init3DSpinner === 'function') {
                    try {
                        init3DSpinner('spinner3d');
                    } catch (err) {
                        console.error('3D Spinner initialization error:', err);
                    }
                } else {
                    console.warn('init3DSpinner function not found.');
                }
            }, 50);

            try {
                var res = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                var data = await res.json();

                if (data.success) {
                    var m = data.measurements;

                    // Display BMI and Body Type in measurement cards
                    if (m.metadata) {
                        var meta = m.metadata;
                        
                        // BMI Card
                        safeUpdate('bmiValue', meta.bmi);
                        var bmiCategoryEl = document.getElementById('bmiCategory');
                        if (bmiCategoryEl) {
                            var categoryText = meta.bmi_category.charAt(0).toUpperCase() + meta.bmi_category.slice(1);
                            bmiCategoryEl.textContent = categoryText;
                            bmiCategoryEl.className = 'result-category ' + getBMICategoryClass(meta.bmi_category);
                        }
                        
                        // Body Type Card
                        safeUpdate('bodyType', formatBodyType(meta.body_type));
                        
                        // Height Card (optional - only if uncommented in HTML)
                        if (document.getElementById('heightDisplayCm')) {
                            safeUpdate('heightDisplayCm', meta.height.cm);
                            safeUpdate('heightDisplayIn', meta.height.inches);
                        }
                        
                        // Weight Card (optional - only if uncommented in HTML)
                        if (document.getElementById('weightDisplayKg')) {
                            safeUpdate('weightDisplayKg', meta.weight.kg);
                            safeUpdate('weightDisplayLbs', meta.weight.lbs);
                        }
                    }

                    // Display measurements - using safe update
                    safeUpdate('neckCircCm', m.neck.circumference.cm);
                    safeUpdate('neckCircIn', m.neck.circumference.inches);

                    safeUpdate('chestCircCm', m.chest.circumference.cm);
                    safeUpdate('chestCircIn', m.chest.circumference.inches);

                    safeUpdate('waistCircCm', m.waist.circumference.cm);
                    safeUpdate('waistCircIn', m.waist.circumference.inches);

                    safeUpdate('hipCircCm', m.hip.circumference.cm);
                    safeUpdate('hipCircIn', m.hip.circumference.inches);

                    safeUpdate('shoulderWidthCm', m.shoulder.width.cm);
                    safeUpdate('shoulderWidthIn', m.shoulder.width.inches);

                    safeUpdate('armHandCm', m.arm.hand_to_elbow.cm);
                    safeUpdate('armHandIn', m.arm.hand_to_elbow.inches);

                    safeUpdate('armShoulderCm', m.arm.shoulder_to_elbow.cm);
                    safeUpdate('armShoulderIn', m.arm.shoulder_to_elbow.inches);

                    safeUpdate('armTotalCm', m.arm.total_length.cm);
                    safeUpdate('armTotalIn', m.arm.total_length.inches);

                    results.classList.add('active');
                    
                    // Smooth scroll to results
                    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    error.textContent = data.error || 'An error occurred';
                    error.classList.add('active');
                }
            } catch (err) {
                console.error('Form submission error:', err);
                error.textContent = 'Network error. Please try again.';
                error.classList.add('active');
            }

            // Hide loading
            loading.classList.remove('active');
            btn.disabled = false;
        });
    }
});