from flask import Flask, flash, request, redirect, url_for, render_template,session
import os
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from auth import auth_bp
import numpy as np

# Loading Pneumonia Model
pneumonia_model = load_model('models/pneumonia_model.h5')

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"
app.register_blueprint(auth_bp, url_prefix='/auth')

# Ensure static files load correctly
app.static_folder = 'static'

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

########################### Routing Functions ########################################

@app.route('/')
def home():
    if 'user_id' in session:
        username=session.get('username','User')
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

@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')

            # Preprocess the image
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img / 255.0

            # Predict using the pneumonia model
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0  # Negative for pneumonia
            else:
                pred = 1  # Positive for pneumonia

            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

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
    app.run(debug=True)