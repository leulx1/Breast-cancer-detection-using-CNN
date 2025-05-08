from flask import Flask, flash, request, redirect, url_for, render_template,session,send_file
import os
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from auth import auth_bp
import numpy as np
from admin import admin_bp
from tensorflow.keras import backend as K
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from PyPDF2 import PdfMerger
import tempfile

# Define the Dice loss function (needed to load the model)
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - dice

# Register custom objects
custom_objects = {'dice_loss': dice_loss}
tf.keras.utils.get_custom_objects().update(custom_objects)

# Loading Models
breast_cancer_model = load_model('unet_mammo.keras', custom_objects=custom_objects)  # Load breast cancer model

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.secret_key = "secret key"
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(admin_bp, url_prefix='/admin')

# Ensure static files load correctly
app.static_folder = 'static'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    """Apply CLAHE to a grayscale image"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def process_breast_cancer_image(image_path):
    """Process the uploaded image and return four versions:
    1. Original
    2. Enhanced (CLAHE)
    3. Segmented mask
    4. Overlay (original + mask)
    """
    try:
        # Read original image
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError("Could not read the image")
        
        # Store original dimensions
        original_height, original_width = original_img.shape
        
        # Create enhanced version (CLAHE)
        enhanced_img = apply_clahe(original_img)
        
        # Prepare image for model (resize to 512x512)
        model_input = cv2.resize(enhanced_img, (512, 512))
        model_input = model_input / 255.0
        model_input = np.expand_dims(model_input, axis=-1)
        model_input = np.expand_dims(model_input, axis=0)
        
        # Get segmentation mask
        mask = breast_cancer_model.predict(model_input)
        binary_mask = (mask > 0.7).astype(np.uint8)[0,...,0]
        
        # Resize mask to original dimensions
        binary_mask = cv2.resize(binary_mask, (original_width, original_height))
        
        # Create segmented version (just the mask)
        segmented_img = binary_mask * 255  # Convert to 0-255 scale
        
        # Create overlay (original + mask)
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img.copy()
            
        overlay = original_rgb.copy()
        overlay[binary_mask == 1] = [255, 0, 0]  # Highlight mask in red
        
        # Save all versions
        result_filename = os.path.basename(image_path)
        
        original_path = os.path.join(app.config['RESULTS_FOLDER'], f"original_{result_filename}")
        enhanced_path = os.path.join(app.config['RESULTS_FOLDER'], f"enhanced_{result_filename}")
        segmented_path = os.path.join(app.config['RESULTS_FOLDER'], f"segmented_{result_filename}")
        overlay_path = os.path.join(app.config['RESULTS_FOLDER'], f"overlay_{result_filename}")
        
        cv2.imwrite(original_path, original_img)
        cv2.imwrite(enhanced_path, enhanced_img)
        cv2.imwrite(segmented_path, segmented_img)
        cv2.imwrite(overlay_path, overlay)
        
        return original_path, enhanced_path, segmented_path, overlay_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None, None

def generate_pdf_report(patient_data, image_paths, output_path):
    """Generate a PDF report with patient data and images"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom style for headings
    heading_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    # Create story (content) for PDF
    story = []
    
    # Add title
    story.append(Paragraph("Breast Cancer Analysis Report", heading_style))
    story.append(Spacer(1, 12))
    
    # Add patient information
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_info = [
        ["First Name:", patient_data['firstname']],
        ["Last Name:", patient_data['lastname']],
        ["Patient ID:", patient_data['patient_id']]
    ]
    patient_table = Table(patient_info, colWidths=[1.5*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 24))
    
    # Add images section
    story.append(Paragraph("Image Analysis Results", styles['Heading2']))
    
    # Add each image with caption
    image_captions = [
        ("Original Mammogram", image_paths['original']),
        ("Enhanced Image (CLAHE)", image_paths['enhanced']),
        ("Segmentation Mask", image_paths['segmented']),
        ("Overlay Visualization", image_paths['overlay'])
    ]
    
    for caption, img_path in image_captions:
        story.append(Spacer(1, 12))
        story.append(Paragraph(caption, styles['Heading2']))
        img = Image(img_path, width=5*inch, height=4*inch)
        story.append(img)
    
    # Build the PDF
    doc.build(story)
########################### Routing Functions ########################################

@app.route('/')
def home():
    if 'user_id' in session:
        username = session.get('username', 'User')
        role = session.get('role', 'radiologist')  # Default to radiologist if role not set
        
        # Check role and render appropriate template
        if role == 'admin':
            return render_template('admin/index.html', username=username)
        else:
            return render_template('homepage.html', username=username)
    return redirect(url_for('auth.login'))


@app.route('/covid.html')
def covid():
    return render_template('covid.html')

@app.route('/admin/index.html')
def admin():
    return render_template('/admin/index.html')

@app.route('/createaccount.html')
def createaccount():
    return render_template('/createaccount.html')
    
@app.route('/manage_accounts.html')
def manage_accounts():
    return render_template('/manage_accounts.html')

@app.route('/services.html')
def services():
    return render_template('services.html')


@app.route('/breastcancer.html')
def brain_tumor():
    return render_template('breastcancer.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/login.html')
def login():
    return render_template('login.html')

########################### Result Functions ########################################

@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        patient_id = request.form['patient_id']
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and get all four versions
            original_path, enhanced_path, segmented_path, overlay_path = process_breast_cancer_image(filepath)
            
            if all([original_path, enhanced_path, segmented_path, overlay_path]):
                return render_template('resultbc.html', 
                                    fn=firstname,
                                    ln=lastname,
                                    pid=patient_id,
                                    original_img=os.path.basename(original_path),
                                    enhanced_img=os.path.basename(enhanced_path),
                                    segmented_img=os.path.basename(segmented_path),
                                    overlay_img=os.path.basename(overlay_path))
            else:
                flash('Error processing the image. Please try again.')
                return redirect(url_for('brain_tumor'))
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(url_for('brain_tumor'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if request.method == 'POST':
        # Get patient data from form
        patient_data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'patient_id': request.form['patient_id']
        }
        
        # Get image filenames from form
        image_filenames = {
            'original': request.form['original_img'],
            'enhanced': request.form['enhanced_img'],
            'segmented': request.form['segmented_img'],
            'overlay': request.form['overlay_img']
        }
        
        # Create full paths to images
        image_paths = {k: os.path.join(app.config['RESULTS_FOLDER'], v) for k, v in image_filenames.items()}
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_path = tmp.name
        
        # Generate the PDF
        generate_pdf_report(patient_data, image_paths, pdf_path)
        
        # Send the PDF to the client
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"BreastCancerReport_{patient_data['lastname']}_{patient_data['patient_id']}.pdf",
            mimetype='application/pdf'
        )

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.run(debug=True)
