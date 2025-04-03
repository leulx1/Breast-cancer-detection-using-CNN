import logging
from flask import Blueprint, request, render_template, redirect, url_for, session, flash, jsonify
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

auth_bp = Blueprint('auth', __name__, static_folder=None)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    logging.debug("Login route accessed")

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        logging.debug(f"Attempting login for email: {email}")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT id, password_hash, name FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            logging.debug(f"Database query executed for email: {email}")

            if user:
                logging.debug(f"User found: {user[0]}")
                if check_password_hash(user[1], password):
                    session['user_id'] = user[0]
                    session['username'] = user[2]
                    flash('Login successful', 'success')
                    logging.info(f"User {user[0]} logged in successfully")
                    return redirect(url_for('home'))
                else:
                    logging.warning(f"Incorrect password for user ID {user[0]}")
                    flash('Invalid credentials', 'danger')
            else:
                logging.warning(f"No user found for email: {email}")
                flash('Invalid credentials', 'danger')

        except Exception as e:
            logging.error(f"Error during login: {e}")
            flash('An error occurred during login', 'danger')
        finally:
            cur.close()
            conn.close()
            logging.debug("Database connection closed")

    return render_template('login.html')

@auth_bp.route('/check_email', methods=['POST'])
def check_email():
    """AJAX route to check if email exists in the database."""
    email = request.form['email']
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
        exists = cur.fetchone()[0] > 0
        return jsonify({'exists': exists})
    except Exception as e:
        logging.error(f"Error checking email: {e}")
        return jsonify({'error': 'Database error'}), 500
    finally:
        cur.close()
        conn.close()

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    logging.debug("Register route accessed")

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        logging.debug(f"Attempting registration for email: {email}")

        hashed_password = generate_password_hash(password)
        logging.debug(f"Password hashed for email: {email}")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            # Check if email already exists before inserting
            cur.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            if cur.fetchone()[0] > 0:
                return jsonify({'success': False, 'message': 'Email already registered!'}), 400

            cur.execute(
                "INSERT INTO users (email, password_hash, name, role) VALUES (%s, %s, %s, %s)", 
                (email, hashed_password, name, role)
            )
            conn.commit()
            logging.info(f"User {email} registered successfully")

            return jsonify({'success': True, 'message': 'Account created successfully!'})

        except psycopg2.IntegrityError:
            conn.rollback()
            logging.warning(f"Email {email} already exists in the database")
            return jsonify({'success': False, 'message': 'Email already registered!'}), 400

        except Exception as e:
            logging.error(f"Error during registration: {e}")
            return jsonify({'success': False, 'message': 'An error occurred while creating your account.'}), 500

        finally:
            cur.close()
            conn.close()
            logging.debug("Database connection closed after registration")

    return render_template('createaccount.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('auth.login'))