from flask import Flask,g, flash, request, redirect, url_for, render_template,session,send_file,make_response,jsonify,current_app,send_from_directory
from extension import mail
from psycopg2.extras import RealDictCursor
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
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from functools import wraps
from auth import login_manager
from models import User
from datetime import datetime
import logging
from database import get_db_connection


def nocache(view):
    @wraps(view)
    def no_cache_view(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return no_cache_view

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
breast_cancer_model=load_model('unet_mammo1.keras', custom_objects=custom_objects)  # Load breast cancer model

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = "secret key"
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(admin_bp, url_prefix='/admin')

login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # This tells Flask-Login where to redirect if not logged in
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id) 
# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.office365.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = ''          #Your email
app.config['MAIL_PASSWORD'] = ''  # your email password and Use app password if 2FA is on
app.config['MAIL_DEFAULT_SENDER'] = ''    #Your email
mail.init_app(app)

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
    3. Segmented mask (blank if no tumor found)
    4. Overlay (original + mask, no overlay if no tumor found)
    Also returns whether tumor found based on 5% threshold.
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
        binary_mask = (mask > 0.5).astype(np.uint8)[0,...,0]
        
        # Resize mask to original dimensions
        binary_mask = cv2.resize(binary_mask, (original_width, original_height))
        
        # Calculate tumor percentage
        total_pixels = original_img.size
        tumor_pixels = np.count_nonzero(binary_mask)
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        print(f"Tumor percentage: {tumor_percentage:.2f}%")
        
        # Determine tumor status
        tumor_found = tumor_percentage > 3  # Boolean for easier logic
        
        # Create blank mask if no tumor found
        if not tumor_found:
            binary_mask = np.zeros_like(binary_mask)
        
        # Create segmented version (just the mask)
        segmented_img = binary_mask * 255  # Convert to 0-255 scale
        
        # Create overlay (original + mask)
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img.copy()
        overlay = original_rgb.copy()
        
        # Only apply overlay if tumor was found
        if tumor_found:
            overlay[binary_mask == 1] = [0, 0, 255]  # Highlight mask in red
        
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
        
        # Return text status based on tumor_found boolean
        status_text = "Cancerous Tumor Found" if tumor_found else "No Cancerous Tumor Found"
        return original_path, enhanced_path, segmented_path, overlay_path, status_text
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None, None, None

def generate_pdf_report(patient_data, image_paths, tumor_status, output_path):
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
    story.append(Paragraph("Breast Cancer Detection Report", heading_style))
    story.append(Spacer(1, 12))
    
    # Add patient information
    story.append(Paragraph("Patient Information", styles['Heading2']))
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_info = [
        ["First Name:", patient_data['firstname']],
        ["Last Name:", patient_data['lastname']],
        ["Patient ID:", patient_data['patient_id']],
        ["Report Generated:", report_time],
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
    story.append(Paragraph("Image Detection Results", styles['Heading2']))
    
    # Add each image with caption
    image_captions = [
        ("Original Mammogram", image_paths['original']),
        ("Enhanced Image (CLAHE)", image_paths['enhanced']),
        ("Suspected areas", image_paths['segmented']),
        ("Overlay Visualization", image_paths['overlay'])
    ]

    for caption, img_path in image_captions:
        story.append(Spacer(1, 12))
        story.append(Paragraph(caption, styles['Heading2']))
        img = Image(img_path, width=5*inch, height=4*inch)
        story.append(img)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Tumor Status", styles['Heading2']))
    
    # Customize tumor status display based on result
    if "not found" in tumor_status.lower() or "no tumor" in tumor_status.lower():
        # Green box for no tumor found
        tumor_style = TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.lightgreen),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.darkgreen),
            ('BOX', (0,0), (-1,-1), 1, colors.darkgreen),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 14),
        ])
    else:
        # Red box for tumor found
        tumor_style = TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.pink),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.darkred),
            ('BOX', (0,0), (-1,-1), 1, colors.darkred),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 14),
        ])
    
    # Create a table with one cell for the tumor status
    tumor_table = Table([[tumor_status]], colWidths=[5*inch], rowHeights=[0.5*inch])
    tumor_table.setStyle(tumor_style)
    story.append(tumor_table)

    # Build the PDF
    doc.build(story)
########################### Routing Functions ########################################

@app.route('/')
@login_required
def home():
    if current_user.role == 'admin':
        return render_template('admin/index.html', username=current_user.name)
    else:
        return render_template('homepage.html', username=current_user.name)



@app.route('/admin/index.html')
@nocache
@login_required
def admin():
    return render_template('/admin/index.html')

@app.route('/createaccount.html')
@nocache
@login_required
def createaccount():
    return render_template('/createaccount.html',username=current_user.name)
    
@app.route('/manage_accounts.html')
@nocache
@login_required
def manage_accounts():
    return render_template('/manage_accounts.html',username=current_user.name)

@app.route('/services.html')
@login_required
@nocache
def services():
    return render_template('services.html', username=current_user.name)


@app.route('/breastcancer.html')
@login_required
@nocache
def brain_tumor():
    return render_template('breastcancer.html', username=current_user.name)

@app.route('/contact.html')
@login_required
@nocache
def contact():
    return render_template('contact.html', username=current_user.name)

@app.route('/about.html')
@login_required
@nocache
def about():
    return render_template('about.html', username=current_user.name)

@app.route('/change_profile.html')
@login_required
@nocache
def change_profile():
    return render_template('change_profile.html', username=current_user.name)

@app.route('/navbar.html')
@login_required
def navbar():
    return render_template('navbar.html', username=current_user.name)

@app.route('/download_pdf/<path:filename>')
@login_required
def download_pdf(filename):
    try:
        # Extract the base file name from the filename parameter
        file_name = os.path.basename(filename)  # e.g., 'BreastCancerReport_fd_34465757.pdf'
        
        # Construct the full path to the file
        full_path = os.path.join(current_app.root_path, 'static', 'reports', file_name)
        full_path = os.path.normpath(full_path)
        
        # Check if the file exists
        if not os.path.exists(full_path):
            current_app.logger.error(f"[ERROR] File not found: {full_path}")
            flash("File not found.", "error")
            return redirect(url_for('previousdetection'))
        
        # Serve the file from the 'static/reports' directory
        return send_from_directory(
            directory=os.path.join(current_app.root_path, 'static', 'reports'),
            path=file_name,
            as_attachment=True
        )
    except Exception as e:
        current_app.logger.error(f"[ERROR] Error serving PDF: {str(e)}")
        flash(f"Error downloading file: {str(e)}", 'error')
        return redirect(url_for('previousdetection'))

@app.route('/previousdetection.html')
@login_required
def previousdetection():
    try:
        print(f"[INFO] current_user.id: {current_user.id}")
        print(f"[INFO] current_user.name: {current_user.name}")
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT DISTINCT ON (pdf_location) *
            FROM previous_detection
            WHERE created_by = %s
            ORDER BY pdf_location, created_at DESC
        """
        print("[INFO] Executing query...")
        cursor.execute(query, (current_user.id,))
        detections = cursor.fetchall()
        
        print(f"[INFO] Raw detections fetched: {detections}")

        formatted_detections = []
        for detection in detections:
            # Normalize pdf_location to use forward slashes
            pdf_location = detection['pdf_location'].replace('\\', '/')
            formatted = {
                'patient_id': detection.get('patient_id', f"Patient {detection['patient_id']}"),
                'pdf_location': pdf_location,  # Use normalized path
                'tumor_status': detection['tumor_status'],
                'created_at': detection['created_at'].strftime('%Y-%m-%d %H:%M') if detection['created_at'] else 'N/A'
            }
            print(f"[INFO] Formatted detection: {formatted}")
            formatted_detections.append(formatted)
        
        return render_template('previousdetection.html', 
                               detections=formatted_detections,
                               username=current_user.name)
        
    except Exception as e:
        current_app.logger.error(f"[ERROR] Database error: {str(e)}")
        flash(f"Error retrieving detections: {str(e)}", 'error')
        return render_template('previousdetection.html', 
                               detections=None,
                               username=current_user.name)



@app.route('/login.html')
def login():
    return render_template('login.html')

########################### Result Functions ########################################

@app.route('/resultbc', methods=['POST'])
def resultbc():
    firstname = request.form.get('firstname')
    lastname = request.form.get('lastname')
    patient_id = request.form.get('patient_id')
    original = request.form.get('original')
    enhanced = request.form.get('enhanced')
    segmented = request.form.get('segmented')
    overlay = request.form.get('overlay')
    status = request.form.get('status')

    return render_template('resultbc.html',
                           fn=firstname,
                           ln=lastname,
                           pid=patient_id,
                           original_img=os.path.basename(original),
                           enhanced_img=os.path.basename(enhanced),
                           segmented_img=os.path.basename(segmented),
                           overlay_img=os.path.basename(overlay),
                           tumor_status=status)

@app.route('/resultfordetection', methods=['POST'])
def resultfordetection():
    try:
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        patient_id = request.form['patient_id']
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            original_path, enhanced_path, segmented_path, overlay_path, tumor_status = process_breast_cancer_image(filepath)
            
            return jsonify({
                'original_path': original_path,
                'enhanced_path': enhanced_path,
                'segmented_path': segmented_path,
                'overlay_path': overlay_path,
                'status_text': tumor_status,
                'tumor_found': tumor_status.lower() != "no cancerous tumor found"
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()  # Logs error to the console
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    # Start logging
    app.logger.info(f"Report generation started at {datetime.now()}")
    app.logger.info(f"Request form data: {request.form}")
    
    if request.method == 'POST':
        try:
            # Validate required fields
            required_fields = [
                'firstname', 'lastname', 'patient_id',
                'original_img', 'enhanced_img', 'segmented_img', 'overlay_img','tumor_status'
            ]
            missing_fields = [field for field in required_fields if field not in request.form]
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                app.logger.error(error_msg)
                return error_msg, 400

            # Get patient data from form
            patient_data = {
                'firstname': request.form['firstname'],
                'lastname': request.form['lastname'],
                'patient_id': request.form['patient_id']
            }
            app.logger.info(f"Processing report for patient: {patient_data}")
            
            # Get image filenames and tumor status from form
            image_filenames = {
                'original_img': request.form['original_img'],
                'enhanced_img': request.form['enhanced_img'],
                'segmented_img': request.form['segmented_img'],
                'overlay_img': request.form['overlay_img']
            }
            tumor_status = request.form['tumor_status']
            app.logger.info(f"Image filenames received: {image_filenames}")
            app.logger.info(f"Tumor status: {tumor_status}")
            
            # Create full paths to images
            image_paths = {
                'original': os.path.join(app.config['RESULTS_FOLDER'], image_filenames['original_img']),
                'enhanced': os.path.join(app.config['RESULTS_FOLDER'], image_filenames['enhanced_img']),
                'segmented': os.path.join(app.config['RESULTS_FOLDER'], image_filenames['segmented_img']),
                'overlay': os.path.join(app.config['RESULTS_FOLDER'], image_filenames['overlay_img'])
            }
            app.logger.info(f"Full image paths: {image_paths}")
            
            # Verify all images exist
            for img_type, path in image_paths.items():
                if not os.path.exists(path):
                    app.logger.error(f"Image not found: {path}")
                    return f"{img_type} image not found", 400
            
            # Create reports directory if it doesn't exist
            reports_dir = app.config['REPORTS_FOLDER']
            os.makedirs(reports_dir, exist_ok=True)
            app.logger.info(f"Reports directory: {reports_dir}")
            
            # Generate PDF filename and path
            pdf_filename = f"BreastCancerReport_{patient_data['lastname']}_{patient_data['patient_id']}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_filename)
            app.logger.info(f"PDF will be saved to: {pdf_path}")
            
            # Generate the PDF
            app.logger.info("Starting PDF generation...")
            generate_pdf_report(patient_data, image_paths, tumor_status, pdf_path)
            app.logger.info("PDF generation completed successfully")
            
            # Save to database
            app.logger.info("Starting database operation...")
            conn = get_db_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO previous_detection 
                    (patient_id, pdf_location, tumor_status, created_by)
                    VALUES (%s, %s, %s, %s)
                """, (
                    patient_data['patient_id'],
                    os.path.join('reports', pdf_filename),
                    tumor_status,
                    current_user.id
                ))
                conn.commit()
                app.logger.info("Database record created successfully")
            except Exception as e:
                conn.rollback()
                app.logger.error(f"Database error: {str(e)}", exc_info=True)
                return "Database error", 500
            finally:
                cur.close()
                conn.close()
            
            # Return the PDF file directly
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
            
        except KeyError as e:
            error_msg = f"Missing required field: {str(e)}"
            app.logger.error(error_msg, exc_info=True)
            return error_msg, 400
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            app.logger.error(error_msg, exc_info=True)
            return error_msg, 500
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
    with app.app_context():
        session.clear()
    # Create results directory if it doesn't exist
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.run(debug=True, use_reloader=True)
