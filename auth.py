from flask import Blueprint, request, render_template, redirect, url_for, session, flash
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            flash('Login successful', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)", 
                        (email, hashed_password, name))
            conn.commit()
            flash('Account created! You can log in now.', 'success')
            return redirect(url_for('auth.login'))
        except psycopg2.IntegrityError:
            flash('Email already registered!', 'danger')
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('auth.login'))
