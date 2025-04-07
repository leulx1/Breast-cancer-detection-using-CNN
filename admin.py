# admin.py
from flask import Blueprint, render_template, session, redirect, url_for, flash

# Create the admin Blueprint
admin_bp = Blueprint('admin', __name__, template_folder='templates')

# Add admin route protection (before_request)
@admin_bp.before_request
def restrict_to_admin():
    if 'role' not in session or session['role'] != 'admin':
        flash('Access denied: Admin privileges required', 'danger')
        return redirect(url_for('auth.login'))

# Admin dashboard route
@admin_bp.route('/')
def index():
    return render_template('admin/index.html')

# You can add more admin routes below...