import logging
from flask import Blueprint, request, render_template, redirect, url_for, session, flash, jsonify
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection
from functools import wraps
# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

auth_bp = Blueprint('auth', __name__, static_folder=None)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        if session.get('role') != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    session.clear()

    logging.debug("Login route accessed")

    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        logging.debug(f"Attempting login for email: {email}")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            # Include role in the SELECT query
            cur.execute("SELECT id, password_hash, name, role FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            logging.debug(f"Database query executed for email: {email}")

            if user:
                logging.debug(f"User found: {user[0]}")
                if check_password_hash(user[1], password):
                    session['user_id'] = user[0]
                    session['username'] = user[2]
                    session['role'] = user[3]
                    print("Session username:", session.get('username'))  
                    logging.info(f"User {user[0]} logged in successfully")
                    
                    
                    if user[3] == 'radiologist':
                        return redirect(url_for('home'))
                    elif user[3] == 'admin':
                        return redirect(url_for('admin.index'))
                    else:
                        return redirect(url_for('home'))
                else:
                    logging.warning(f"Incorrect password for user ID {user[0]}")
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

@auth_bp.route('/manage_accounts')
@admin_required
def manage_accounts():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, email, name, password_hash, role FROM users")
        rows = cur.fetchall()
        users = [{'id': r[0], 'email': r[1], 'name': r[2], 'role': r[4]} for r in rows]
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        flash("Failed to load user list.", "danger")
        users = []
    finally:
        cur.close()
        conn.close()

    return render_template('manage_accounts.html', users=users)




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

        return render_template('update_user.html', user={
            'id': user[0], 'name': user[1], 'email': user[2], 'role': user[3]
        })




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

@auth_bp.route('/get_user_counts')
@admin_required
def get_user_counts():
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get count of radiologists
        cur.execute("SELECT COUNT(*) FROM users WHERE role = 'radiologist'")
        radiologist_count = cur.fetchone()[0]
        
        # Get count of admins
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

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('auth.login'))