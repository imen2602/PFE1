from flask import Flask, render_template, request, jsonify
import os
from matcher import get_engine
from preprocessing import extract_text_from_pdf

app = Flask(__name__)

# Configuration du dossier pour stocker les CV téléchargés
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Affiche la page d'accueil de la plateforme."""
    return render_template('index.html')

@app.route('/api/candidate', methods=['POST'])
def candidate_space():
    """Gère l'espace candidat en analysant le CV téléchargé pour suggérer des offres."""
    if 'cv_file' not in request.files:
        return jsonify({"error": "Aucun fichier CV"}), 400
    
    file = request.files['cv_file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Format non supporté, PDF uniquement"}), 400
        
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    cv_text = extract_text_from_pdf(path)
    engine = get_engine()
    results = engine.match_cv_to_jobs(cv_text)
    
    return jsonify({"results": results})

@app.route('/api/recruiter', methods=['POST'])
def recruiter_space():
    """Gère l'espace recruteur en cherchant les CV les plus adaptés à une offre."""
    job_desc = request.form.get('job_description', '')
    if not job_desc:
        return jsonify({"error": "Description vide"}), 400
    
    engine = get_engine()
    results = engine.match_job_to_cvs(job_desc)
    
    return jsonify({"results": results})

if __name__ == '__main__':
    print("Démarrage de RecrutIA (Modèle BERT Multilingue)...")
    get_engine()
    app.run(debug=True, port=5000)
