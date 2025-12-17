// 3D Smooth Human Mesh Spinner - Realistic Body
// Save as: static/js/spinner3d.js

function init3DSpinner(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Container not found:', containerId);
        return;
    }

    // Clear any existing content
    container.innerHTML = '';

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(180, 180);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Enhanced lighting for smooth appearance
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
    keyLight.position.set(5, 10, 7);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(0, 5, -10);
    scene.add(backLight);

    // Smooth white material
    const bodyMaterial = new THREE.MeshPhongMaterial({ 
        color: 0xffffff,
        shininess: 8,
        flatShading: false,
        side: THREE.FrontSide
    });

    // Very subtle wireframe overlay
    const wireMaterial = new THREE.MeshBasicMaterial({ 
        color: 0xdddddd,
        wireframe: true,
        transparent: true,
        opacity: 0.15
    });

    const bodyGroup = new THREE.Group();

    // Create smooth human body using more detailed spheres and cylinders
    
    // HEAD - smooth sphere
    const headGeo = new THREE.SphereGeometry(0.28, 32, 32);
    const head = new THREE.Mesh(headGeo, bodyMaterial);
    head.position.y = 1.55;
    head.scale.set(1, 1.1, 1);
    bodyGroup.add(head);
    const headWire = new THREE.Mesh(headGeo, wireMaterial);
    head.add(headWire);

    // NECK - smooth connection
    const neckGeo = new THREE.CylinderGeometry(0.12, 0.14, 0.2, 24);
    const neck = new THREE.Mesh(neckGeo, bodyMaterial);
    neck.position.y = 1.32;
    bodyGroup.add(neck);
    const neckWire = new THREE.Mesh(neckGeo, wireMaterial);
    neck.add(neckWire);

    // UPPER TORSO - smooth chest
    const chestGeo = new THREE.SphereGeometry(0.38, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.55);
    const chest = new THREE.Mesh(chestGeo, bodyMaterial);
    chest.position.y = 1.15;
    chest.scale.set(1, 1.2, 0.8);
    bodyGroup.add(chest);
    const chestWire = new THREE.Mesh(chestGeo, wireMaterial);
    chest.add(chestWire);

    // MID TORSO - smooth transition
    const midTorsoGeo = new THREE.CylinderGeometry(0.36, 0.34, 0.35, 32);
    const midTorso = new THREE.Mesh(midTorsoGeo, bodyMaterial);
    midTorso.position.y = 0.72;
    bodyGroup.add(midTorso);
    const midTorsoWire = new THREE.Mesh(midTorsoGeo, wireMaterial);
    midTorso.add(midTorsoWire);

    // WAIST - narrow section
    const waistGeo = new THREE.CylinderGeometry(0.34, 0.32, 0.25, 32);
    const waist = new THREE.Mesh(waistGeo, bodyMaterial);
    waist.position.y = 0.42;
    bodyGroup.add(waist);
    const waistWire = new THREE.Mesh(waistGeo, wireMaterial);
    waist.add(waistWire);

    // HIPS - smooth pelvis area
    const hipsGeo = new THREE.SphereGeometry(0.35, 32, 24, 0, Math.PI * 2, Math.PI * 0.45, Math.PI * 0.55);
    const hips = new THREE.Mesh(hipsGeo, bodyMaterial);
    hips.position.y = 0.15;
    hips.scale.set(1.1, 0.8, 1);
    bodyGroup.add(hips);
    const hipsWire = new THREE.Mesh(hipsGeo, wireMaterial);
    hips.add(hipsWire);

    // SHOULDERS - smooth rounded joints
    function createShoulder(x) {
        const shoulderGeo = new THREE.SphereGeometry(0.12, 24, 24);
        const shoulder = new THREE.Mesh(shoulderGeo, bodyMaterial);
        shoulder.position.set(x, 1.15, 0);
        bodyGroup.add(shoulder);
        const shoulderWire = new THREE.Mesh(shoulderGeo, wireMaterial);
        shoulder.add(shoulderWire);
        return shoulder;
    }
    createShoulder(-0.44);
    createShoulder(0.44);

    // ARMS - smooth limbs
    function createArm(x) {
        // Upper arm
        const upperArmGeo = new THREE.CylinderGeometry(0.09, 0.08, 0.42, 24);
        const upperArm = new THREE.Mesh(upperArmGeo, bodyMaterial);
        upperArm.position.set(x, 0.85, 0);
        bodyGroup.add(upperArm);
        const upperArmWire = new THREE.Mesh(upperArmGeo, wireMaterial);
        upperArm.add(upperArmWire);

        // Elbow
        const elbowGeo = new THREE.SphereGeometry(0.08, 20, 20);
        const elbow = new THREE.Mesh(elbowGeo, bodyMaterial);
        elbow.position.set(x, 0.62, 0);
        bodyGroup.add(elbow);
        const elbowWire = new THREE.Mesh(elbowGeo, wireMaterial);
        elbow.add(elbowWire);

        // Forearm
        const forearmGeo = new THREE.CylinderGeometry(0.08, 0.07, 0.38, 24);
        const forearm = new THREE.Mesh(forearmGeo, bodyMaterial);
        forearm.position.set(x, 0.38, 0);
        bodyGroup.add(forearm);
        const forearmWire = new THREE.Mesh(forearmGeo, wireMaterial);
        forearm.add(forearmWire);

        // Hand
        const handGeo = new THREE.SphereGeometry(0.07, 20, 20);
        handGeo.scale(1, 1.3, 0.7);
        const hand = new THREE.Mesh(handGeo, bodyMaterial);
        hand.position.set(x, 0.16, 0);
        bodyGroup.add(hand);
        const handWire = new THREE.Mesh(handGeo, wireMaterial);
        hand.add(handWire);
    }
    createArm(-0.44);
    createArm(0.44);

    // LEGS - smooth limbs
    function createLeg(x) {
        // Upper leg
        const upperLegGeo = new THREE.CylinderGeometry(0.13, 0.11, 0.48, 24);
        const upperLeg = new THREE.Mesh(upperLegGeo, bodyMaterial);
        upperLeg.position.set(x, -0.16, 0);
        bodyGroup.add(upperLeg);
        const upperLegWire = new THREE.Mesh(upperLegGeo, wireMaterial);
        upperLeg.add(upperLegWire);

        // Knee
        const kneeGeo = new THREE.SphereGeometry(0.11, 20, 20);
        const knee = new THREE.Mesh(kneeGeo, bodyMaterial);
        knee.position.set(x, -0.42, 0);
        bodyGroup.add(knee);
        const kneeWire = new THREE.Mesh(kneeGeo, wireMaterial);
        knee.add(kneeWire);

        // Lower leg
        const lowerLegGeo = new THREE.CylinderGeometry(0.11, 0.09, 0.48, 24);
        const lowerLeg = new THREE.Mesh(lowerLegGeo, bodyMaterial);
        lowerLeg.position.set(x, -0.70, 0);
        bodyGroup.add(lowerLeg);
        const lowerLegWire = new THREE.Mesh(lowerLegGeo, wireMaterial);
        lowerLeg.add(lowerLegWire);

        // Ankle
        const ankleGeo = new THREE.SphereGeometry(0.09, 16, 16);
        const ankle = new THREE.Mesh(ankleGeo, bodyMaterial);
        ankle.position.set(x, -0.96, 0);
        bodyGroup.add(ankle);
        const ankleWire = new THREE.Mesh(ankleGeo, wireMaterial);
        ankle.add(ankleWire);

        // Foot
        const footGeo = new THREE.SphereGeometry(0.09, 20, 20);
        footGeo.scale(0.9, 0.6, 1.4);
        const foot = new THREE.Mesh(footGeo, bodyMaterial);
        foot.position.set(x, -1.06, 0.06);
        bodyGroup.add(foot);
        const footWire = new THREE.Mesh(footGeo, wireMaterial);
        foot.add(footWire);
    }
    createLeg(-0.16);
    createLeg(0.16);

    scene.add(bodyGroup);

    // Position camera - zoomed out to show full body including head
    camera.position.set(0, 0.3, 3.8);
    camera.lookAt(0, 0.3, 0);

    // Animation - smooth auto-rotation
    let rotationSpeed = 0.015;
    let time = 0;
    let isDragging = false;
    let autoRotate = true;
    let animationId;

    function animate() {
        animationId = requestAnimationFrame(animate);
        time += 0.016;
        
        // Auto-rotate continuously
        if (autoRotate && !isDragging) {
            bodyGroup.rotation.y += rotationSpeed;
        }
        
        // Very subtle floating
        bodyGroup.position.y = Math.sin(time * 1.0) * 0.02;
        
        renderer.render(scene, camera);
    }

    animate();

    // Interaction
    let prevX = 0, prevY = 0;

    const startDrag = (x, y) => {
        isDragging = true;
        autoRotate = false;
        prevX = x;
        prevY = y;
    };

    const drag = (x, y) => {
        if (isDragging) {
            bodyGroup.rotation.y += (x - prevX) * 0.01;
            bodyGroup.rotation.x += (y - prevY) * 0.01;
            prevX = x;
            prevY = y;
        }
    };

    const endDrag = () => {
        isDragging = false;
        setTimeout(() => { autoRotate = true; }, 1000);
    };

    container.addEventListener('mousedown', (e) => startDrag(e.clientX, e.clientY));
    document.addEventListener('mousemove', (e) => drag(e.clientX, e.clientY));
    document.addEventListener('mouseup', endDrag);

    container.addEventListener('touchstart', (e) => {
        if (e.touches[0]) startDrag(e.touches[0].clientX, e.touches[0].clientY);
    });
    document.addEventListener('touchmove', (e) => {
        if (e.touches[0]) drag(e.touches[0].clientX, e.touches[0].clientY);
    });
    document.addEventListener('touchend', endDrag);

    console.log('3D Human Mesh Spinner initialized!');

    return () => {
        cancelAnimationFrame(animationId);
        renderer.dispose();
        container.innerHTML = '';
    };
}

console.log('spinner3d.js loaded - init3DSpinner ready');