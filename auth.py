import logging
from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from functools import wraps
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection
from flask_mail import Message
import smtplib
from extension import mail
import random
import string
import base64
import re
from models import User

# Initialize blueprint
auth_bp = Blueprint('auth', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Setup Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(auth_bp)


# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        if current_user.role != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


# Generate secure password
def generate_secure_password():
    while True:
        password = ''.join(random.choices(
            string.ascii_letters + string.digits + "!@#$%^&*()", k=8
        ))
        if (any(c.isupper() for c in password) and 
            any(c.isdigit() for c in password) and 
            any(c in "!@#$%^&*()" for c in password)):
            return password


# Login route
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    logging.debug("Login route accessed")
    if request.method == 'POST':
        session.clear()
        email = request.form['email'].lower()
        password = request.form['password']
        logging.debug(f"Attempting login for email: {email}")
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, password_hash, name, role FROM users WHERE email = %s", (email,))
            user_data = cur.fetchone()
            logging.debug(f"Database query executed for email: {email}")

            if user_data:
                logging.debug(f"User found: {user_data[0]}")
                if check_password_hash(user_data[1], password):
                    # Create User object and log in
                    user_obj = User(id=user_data[0], name=user_data[2], email=email, role=user_data[3])
                    login_user(user_obj)

                    flash('Logged in successfully.', 'success')

                    # Role-based redirection
                    if user_obj.role == 'radiologist':
                        return redirect(url_for('home'))
                    elif user_obj.role == 'admin':
                        return redirect(url_for('admin.index'))  # Make sure this route exists
                    else:
                        return redirect(url_for('home'))
                else:
                    logging.warning(f"Incorrect password for user ID {user_data[0]}")
                    flash('Wrong Email or Password please try again!!!', 'login_error')
            else:
                logging.warning(f"No user found for email: {email}")
                flash('Wrong Email or Password please try again!!!', 'login_error')
        except Exception as e:
            logging.error(f"Error during login: {e}")
            flash('An error occurred during login', 'login_error')
        finally:
            cur.close()
            conn.close()
            logging.debug("Database connection closed")
    return render_template('login.html')

# Check email existence
@auth_bp.route('/check_email', methods=['POST'])
@admin_required
def check_email():
    email = request.form.get('email').lower()
    user_id = request.form.get('user_id')  # this can be None if not sent
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if user_id:
            cur.execute("SELECT id FROM users WHERE email = %s AND id != %s", (email, user_id))
        else:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()
        return jsonify({'exists': bool(existing_user)})
    finally:
        cur.close()
        conn.close()


# Register route
@auth_bp.route('/register', methods=['GET', 'POST'])
@admin_required
def register():
    logging.debug("Register route accessed")
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email'].lower()
        password = request.form['password']
        role = request.form['role']
        logging.debug(f"Attempting registration for email: {email}")
        hashed_password = generate_password_hash(password)
        logging.debug(f"Password hashed for email: {email}")
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            if cur.fetchone()[0] > 0:
                return jsonify({'success': False, 'message': 'Email already registered!'}), 400

            # Only send email if role is radiologist
            if role.lower() == 'radiologist':
                msg = Message('Your Account Details', recipients=[email])
                msg.body = f"""
                Hello {name},Welcome to Selale Comphrensive Hospital Breast Cancer Detection System!!!
                Your account has been created successfully!
                Here are your login credentials:
                Email: {email}
                Password: {password}
                """
                try:
                    mail.send(msg)
                    logging.info(f"Email successfully sent to {email}")
                except smtplib.SMTPRecipientsRefused:
                    logging.error(f"Email sending failed: recipient {email} not found.")
                    return jsonify({'success': False, 'message': 'Email not found. Please check the email address.'}), 400
                except Exception as e:
                    logging.error(f"Email sending failed for {email}: {e}")
                    return jsonify({'success': False, 'message': 'Email could not be sent.'}), 500

            cur.execute(
                "INSERT INTO users (email, password_hash, name, role) VALUES (%s, %s, %s, %s)",
                (email, hashed_password, name, role)
            )
            conn.commit()
            logging.info(f"User {email} registered successfully")
            message = 'Email sent and account created successfully!' if role.lower() == 'radiologist' else 'Account created successfully!'
            return jsonify({'success': True, 'message': message})
        except Exception as e:
            conn.rollback()
            logging.error(f"Error during registration: {e}")
            return jsonify({'success': False, 'message': 'An error occurred while creating your account.'}), 500
        finally:
            cur.close()
            conn.close()
            logging.debug("Database connection closed after registration")
    return render_template('createaccount.html')


# Forgot password
@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].lower()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        if not user:
            cur.close()
            conn.close()
            return jsonify(message="Email not found. Please check and try again.", success=False)
        new_password = generate_secure_password()
        hashed_password = generate_password_hash(new_password)
        try:
            cur.execute("UPDATE users SET password_hash = %s WHERE email = %s", (hashed_password, email))
            conn.commit()
            msg = Message('Password Reset', recipients=[email])
            msg.body = f"""
            Hello,
            A new password has been generated for your account.
            New Password: {new_password}
            Please log in and change it as soon as possible.
            Regards,
            Your Support Team
            """
            mail.send(msg)
            return jsonify(message="Password successfully changed. Check your email.", success=True)
        except Exception as e:
            conn.rollback()
            return jsonify(message="An error occurred while resetting your password.", success=False), 500
        finally:
            cur.close()
            conn.close()
    return render_template('forgotpassword.html')


# Manage accounts
@auth_bp.route('/manage_accounts')
@admin_required
def manage_accounts():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, email, name, role FROM users")
        rows = cur.fetchall()
        users = [{'id': r[0], 'email': r[1], 'name': r[2], 'role': r[3]} for r in rows]
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        flash("Failed to load user list.", "danger")
        users = []
    finally:
        cur.close()
        conn.close()
    return render_template('manage_accounts.html', users=users)


# Update user
@auth_bp.route('/update_user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def update_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email'].lower()
        role = request.form['role']
        password = request.form.get('password')
        try:
            if password:
                hashed_password = generate_password_hash(password)
                cur.execute(
                    "UPDATE users SET name=%s, email=%s, password_hash=%s, role=%s, updated_at=NOW() WHERE id=%s",
                    (name, email, hashed_password, role, user_id)
                )
            else:
                cur.execute(
                    "UPDATE users SET name=%s, email=%s, role=%s, updated_at=NOW() WHERE id=%s",
                    (name, email, role, user_id)
                )
            conn.commit()
            flash("User updated successfully.", "user-updated")
            return redirect(url_for('auth.manage_accounts'))
        except Exception as e:
            logging.error(f"Error updating user: {e}")
            flash("Failed to update user.", "danger")
        finally:
            cur.close()
            conn.close()
    else:
        cur.execute("SELECT id, name, email, role FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user is None:
            flash("User not found.", "warning")
            return redirect(url_for('auth.manage_accounts'))
        return render_template('update_user.html', user={'id': user[0], 'name': user[1], 'email': user[2], 'role': user[3]})


# Delete user
@auth_bp.route('/delete_user/<int:user_id>', methods=['GET'])
@admin_required
def delete_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        flash("User deleted successfully.", "success")
    except Exception as e:
        logging.error(f"Error deleting user: {e}")
        flash("Failed to delete user.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('auth.manage_accounts'))


# Get user counts
@auth_bp.route('/get_user_counts')
@admin_required
def get_user_counts():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM users WHERE role = 'radiologist'")
        radiologist_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        admin_count = cur.fetchone()[0]
        return jsonify({
            'success': True,
            'radiologist_count': radiologist_count,
            'admin_count': admin_count
        })
    except Exception as e:
        logging.error(f"Error fetching user counts: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch user counts'}), 500
    finally:
        cur.close()
        conn.close()


# Logout route
@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('auth.login'))


# Save to DB route
@auth_bp.route('/save_to_db', methods=['POST'])
@login_required
def save_to_db():
    try:
        data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'patient_id': request.form['patient_id'],
            'overlay_img': request.form['overlay_img'],
            'tumor_status': request.form['tumor_status']
        }
        if not all(data.values()):
            return jsonify({"status": "no", "message": "All fields are required"}), 400

        img_data = data['overlay_img']
        if img_data.startswith('data:image'):
            img_data = img_data.split('base64,')[-1]
        img_data = img_data.strip()
        padding = len(img_data) % 4
        if padding:
            img_data += '=' * (4 - padding)

        try:
            image_binary = base64.b64decode(img_data)
        except Exception as e:
            return jsonify({"status": "no", "message": "Invalid image data - must be properly encoded base64"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO analysis_results 
            (patient_first_name, patient_last_name, patient_id, result_image, tumor_status, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            data['firstname'], data['lastname'], data['patient_id'],
            image_binary, data['tumor_status'], current_user.id
        ))
        conn.commit()
        return jsonify({"status": "yes", "message": "Data saved successfully"})
    except Exception as e:
        if 'conn' in locals(): conn.rollback()
        return jsonify({"status": "no", "message": f"Error: {str(e)}"}), 500
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

@auth_bp.route('/change_profile', methods=['GET', 'POST'])
@login_required
def change_profile():
    if request.method == 'POST':
        old_password = request.form['old_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_new_password']

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('auth.change_profile'))  # or render again

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT id, password_hash FROM users WHERE id = %s", (current_user.id,))
            user_record = cur.fetchone()

            if not user_record or not check_password_hash(user_record[1], old_password):
                flash('Old password is incorrect.', 'danger')
                return redirect(url_for('auth.change_profile'))

            hashed_password = generate_password_hash(new_password)
            cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (hashed_password, current_user.id))
            conn.commit()
            flash('Your password has been successfully changed!', 'success')
            return redirect(url_for('auth.change_profile'))
        except Exception as e:
            conn.rollback()
            flash(f"An error occurred: {str(e)}", 'danger')
        finally:
            cur.close()
            conn.close()

    return render_template('change_profile.html')