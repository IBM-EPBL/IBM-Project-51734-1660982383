from flask import Flask, request, redirect, flash, send_from_directory, render_template, send_file



app = Flask(__name__, static_url_path='/')
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


