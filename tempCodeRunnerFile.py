import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from spectral.io import envi

# Load model and selector once
model = joblib.load('optimized_soil_carbon_model.pkl')
selector = joblib.load('feature_selector.pkl')

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure required folders exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables to store map and image shape
org_c_map = None
rows, cols = 0, 0

# Function to add spectral indices
def add_spectral_indices(X):
    df = X.copy()
    if 'W700' in df.columns and 'W1700' in df.columns:
        df['NDI_700_1700'] = (df['W700'] - df['W1700']) / (df['W700'] + df['W1700'])
    if 'W600' in df.columns and 'W2100' in df.columns:
        df['SR_600_2100'] = df['W600'] / df['W2100']
    if 'W2200' in df.columns:
        df['CR_2200'] = df['W2200'] / (0.5 * (df.get('W2100', 1) + df.get('W2300', 1)))
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    global org_c_map, rows, cols
    value = None

    if request.method == 'POST':
        # Get uploaded files
        hdr_file = request.files['hdr_file']
        bin_file = request.files['bin_file']

        # Secure filenames
        hdr_filename = secure_filename(hdr_file.filename)
        bin_filename = secure_filename(bin_file.filename)

        # Save files
        hdr_path = os.path.join(app.config['UPLOAD_FOLDER'], hdr_filename)
        bin_path = os.path.join(app.config['UPLOAD_FOLDER'], bin_filename)

        hdr_file.save(hdr_path)
        bin_file.save(bin_path)

        # Load image
        img = envi.open(hdr_path, bin_path)
        cube = img.load()  # shape: (rows, cols, bands)

        # Get wavelengths
        wavelengths = img.metadata.get("wavelength")
        wavelengths = [float(w) for w in wavelengths]

        # Prepare data
        rows, cols, bands = cube.shape
        pixel_data = cube.reshape(-1, bands)

        band_names = [f"W{int(w)}" for w in wavelengths]
        pixel_df = pd.DataFrame(pixel_data, columns=band_names)

        # Add spectral indices
        pixel_df = add_spectral_indices(pixel_df)

        # Select features
        selected_columns = selector.get_support(indices=True)
        selected_features = pixel_df.columns[selected_columns]
        X_selected = pixel_df[selected_features]

        # Predict
        y_pred_flat = model.predict(X_selected)
        org_c_map = y_pred_flat.reshape(rows, cols)

        # Save map image
        plt.figure(figsize=(10, 6))
        plt.imshow(org_c_map, cmap='YlGn', interpolation='none')
        plt.title("Predicted Soil Organic Carbon Map")
        plt.colorbar(label='Org C (%)')
        plt.axis('off')
        plt.savefig('static/output_map.png')
        plt.close()

    return render_template('index.html', rows=rows, cols=cols, value=value)

@app.route('/get_value', methods=['POST'])
def get_value():
    global org_c_map, rows, cols

    # Get requested row/col
    row = int(request.form['row'])
    col = int(request.form['col'])

    # Clip values to valid range
    row = max(0, min(row, rows - 1))
    col = max(0, min(col, cols - 1))

    # Get value
    value = org_c_map[row, col] if org_c_map is not None else None

    # Re-render index with value shown
    return render_template('index.html', rows=rows, cols=cols, value=f"{value:.3f} %")

if __name__ == '__main__':
    app.run(debug=True)
