import logging
from flask import Blueprint, request, render_template, redirect, url_for, session, flash
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

auth_bp = Blueprint('auth', __name__)

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
            cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            logging.debug(f"Database query executed for email: {email}")

            if user:
                logging.debug(f"User found: {user[0]}")
                if check_password_hash(user[1], password):
                    session['user_id'] = user[0]
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
        finally:
            cur.close()
            conn.close()
            logging.debug("Database connection closed")

    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    logging.debug("Register route accessed")

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        logging.debug(f"Attempting registration for email: {email}")

        hashed_password = generate_password_hash(password)
        logging.debug(f"Password hashed for email: {email}")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)", 
                        (email, hashed_password, name))
            conn.commit()
            flash('Account created! You can log in now.', 'success')
            logging.info(f"User {email} registered successfully")
            return redirect(url_for('auth.login'))

        except psycopg2.IntegrityError:
            flash('Email already registered!', 'danger')
            conn.rollback()
            logging.warning(f"Email {email} already exists in the database")

        except Exception as e:
            logging.error(f"Error during registration: {e}")

        finally:
            cur.close()
            conn.close()
            logging.debug("Database connection closed after registration")

    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    user_id = session.get('user_id', 'Unknown')
    logging.debug(f"User {user_id} logging out")

    session.pop('user_id', None)
    flash('Logged out successfully', 'success')

    logging.info(f"User {user_id} logged out")
    return redirect(url_for('auth.login'))
