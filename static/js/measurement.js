document.addEventListener('DOMContentLoaded', function () {

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
        });
    });

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

        if (results) results.classList.remove('active');
        if (error) error.classList.remove('active');

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

                // Body Type
                safeUpdate('bodyType', formatBodyType(meta.body_type));

                // ✅ RECOMMENDED SIZE (FIXED)
                safeUpdate('recommendedSize', meta.recommended_size || '-');
            }

            /* ===============================
               Measurements
            =============================== */
            safeUpdate('neckCircCm', m.neck?.circumference?.cm);
            safeUpdate('neckCircIn', m.neck?.circumference?.inches);

            safeUpdate('chestCircCm', m.chest?.circumference?.cm);
            safeUpdate('chestCircIn', m.chest?.circumference?.inches);

            safeUpdate('waistCircCm', m.waist?.circumference?.cm);
            safeUpdate('waistCircIn', m.waist?.circumference?.inches);

            safeUpdate('hipCircCm', m.hip?.circumference?.cm);
            safeUpdate('hipCircIn', m.hip?.circumference?.inches);

            safeUpdate('shoulderWidthCm', m.shoulder?.width?.cm);
            safeUpdate('shoulderWidthIn', m.shoulder?.width?.inches);

            safeUpdate('armHandCm', m.arm?.hand_to_elbow?.cm);
            safeUpdate('armHandIn', m.arm?.hand_to_elbow?.inches);

            safeUpdate('armShoulderCm', m.arm?.shoulder_to_elbow?.cm);
            safeUpdate('armShoulderIn', m.arm?.shoulder_to_elbow?.inches);

            safeUpdate('armTotalCm', m.arm?.total_length?.cm);
            safeUpdate('armTotalIn', m.arm?.total_length?.inches);

            if (results) {
                results.classList.add('active');
                results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

        } catch (err) {
            console.error(err);
            if (error) {
                error.textContent = err.message || 'Network error. Please try again.';
                error.classList.add('active');
            }
        }

        if (loading) loading.classList.remove('active');
        if (btn) btn.disabled = false;
    });
});
