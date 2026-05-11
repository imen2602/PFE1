import numpy as np
import json
import os
import pandas as pd
import faiss
import re
from sentence_transformers import SentenceTransformer
from preprocessing import get_clean_pipeline, build_weighted_text, TECHNICAL_SKILLS_LIST

# Chargement du modèle BERT multilingue pour transformer le texte en vecteurs
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(MODEL_NAME)

# Dictionnaire des mots-clés par domaine pour garantir un filtrage cohérent
DOMAIN_KEYWORDS = {
    'Embarque': {'c++', 'cpp', 'stm32', 'freertos', 'rtos', 'do178', 'trace32', 'vhdl', 'fpga', 'can', 'firmware', 'arm', 'aerospatial', 'avionique'},
    'Informatique': {'java', 'spring', 'mysql', 'python', 'flask', 'sql', 'devops', 'docker', 'kubernetes'},
    'FullStack': {'react', 'node', 'mongodb', 'docker', 'git', 'javascript', 'nodejs'},
    'DataScience': {'machinelearning', 'python', 'scikit', 'datascience', 'deeplearning', 'ai', 'tensorflow', 'ml'},
    'Securite': {'cybersecurite', 'pentest', 'reseaux', 'parefeu', 'vulnerabilites', 'siem', 'soc', 'kalilinux', 'forensics', 'cryptographie', 'ethicalhacking', 'ids', 'ips', 'securite'},
    'Finance': {'finance', 'excel', 'audit', 'comptabilite', 'sage', 'fiscalite', 'sap', 'bilan'},
    'Marketing': {'marketingdigital', 'seo', 'communication', 'reseauxsociaux', 'marketing'},
    'Telecom': {'telecom', 'reseaux', 'cisco', '5g', 'fibre', 'optique'}
}

class MatchingEngine:
    """Moteur de matching intelligent utilisant BERT et la recherche vectorielle FAISS."""
    
    def __init__(self, cv_path, jobs_path):
        """Initialise le moteur en chargeant les CV et les offres puis en créant les index de recherche."""
        self.cvs = []
        if os.path.exists(cv_path):
            with open(cv_path, 'r', encoding='utf-8') as f:
                self.cvs = json.load(f)

        self.jobs = []
        if os.path.exists(jobs_path):
            df = pd.read_csv(jobs_path)
            self.jobs = df.fillna("").to_dict('records')

        # Création de l'index de recherche pour les CV (espace recruteur)
        self.weighted_cv_texts = [
            get_clean_pipeline(build_weighted_text(cv)) for cv in self.cvs
        ]
        emb = model.encode(self.weighted_cv_texts, show_progress_bar=False)
        faiss.normalize_L2(emb)
        self.cv_index = faiss.IndexFlatIP(emb.shape[1])
        self.cv_index.add(emb.astype('float32'))
        self.cv_embeddings = emb

        # Création de l'index de recherche pour les offres (espace candidat)
        self.weighted_job_texts = []
        for job in self.jobs:
            raw = f"{job['title']} {job['description']} {job['skills']}"
            self.weighted_job_texts.append(get_clean_pipeline(raw))
        emb_j = model.encode(self.weighted_job_texts, show_progress_bar=False)
        faiss.normalize_L2(emb_j)
        self.job_index = faiss.IndexFlatIP(emb_j.shape[1])
        self.job_index.add(emb_j.astype('float32'))

    def match_job_to_cvs(self, job_description):
        """Recherche les meilleurs CV correspondant à une description d'offre donnée."""
        processed = get_clean_pipeline(job_description)
        q_emb = model.encode([processed], show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        D, I = self.cv_index.search(q_emb.astype('float32'), len(self.cvs))

        job_lower = job_description.lower()
        results = []
        
        for idx, sem_score in zip(I[0], D[0]):
            sem = max(0.0, float(sem_score))
            cv  = self.cvs[idx]

            # Vérification de la cohérence du domaine
            cv_domain = cv.get('domaine')
            if cv_domain in DOMAIN_KEYWORDS:
                job_words = set(processed.split())
                if not any(w in job_words for w in DOMAIN_KEYWORDS[cv_domain]):
                    continue

            # Vérification du niveau de diplôme pour les postes à responsabilité
            job_title_lower = job_description.lower()
            is_job_senior = any(w in job_title_lower for w in ['ingénieur', 'ingenieur', 'senior', 'manager', 'responsable', 'expert'])
            is_cv_licence = 'licence' in cv['diplome'].lower()
            if is_job_senior and is_cv_licence:
                continue

            # Calcul du score basé sur les mots-clés communs et la sémantique
            matched = [s for s in cv['competences'] if s.lower() in job_lower]
            skill_score = len(matched) / max(len(cv['competences']), 1)
            final = 0.80 * sem + 0.20 * skill_score
            score = min(88.0, max(0.0, round(final * 100, 1)))

            # Ajout d'un bonus basé sur les années d'expérience
            exp_match = re.search(r'(\d+)\s+ans', cv.get('experience', ''))
            exp_years = int(exp_match.group(1)) if exp_match else 0
            score += min(exp_years, 10) * 0.5
            score = min(95.0, score)

            if score >= 30:
                results.append({
                    "item": cv,
                    "score_global": score,
                    "matched_skills": matched if matched else cv['competences'][:2]
                })

        return sorted(results, key=lambda x: x['score_global'], reverse=True)[:5]

    def match_cv_to_jobs(self, raw_cv_text):
        """Recherche les offres d'emploi les plus adaptées au profil d'un candidat."""
        processed = get_clean_pipeline(raw_cv_text)
        q_emb = model.encode([processed], show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        D, I = self.job_index.search(q_emb.astype('float32'), len(self.jobs))

        cv_tech_keywords = set(w for w in processed.split() if w in TECHNICAL_SKILLS_LIST)
        cv_lower = raw_cv_text.lower()
        is_cv_licence_only = 'licence' in cv_lower and 'ingénieur' not in cv_lower and 'ingenieur' not in cv_lower and 'master' not in cv_lower

        results = []
        for idx, sem_score in zip(I[0], D[0]):
            sem = max(0.0, float(sem_score))
            job = self.jobs[idx]
            job_text = self.weighted_job_texts[idx]
            
            # Filtrage par domaine
            job_domain = job.get('domaine')
            if job_domain in DOMAIN_KEYWORDS:
                if not any(w in cv_tech_keywords for w in DOMAIN_KEYWORDS[job_domain]):
                    continue
                
            # Filtrage par niveau d'étude
            job_title_lower = job['title'].lower()
            is_job_senior = any(w in job_title_lower for w in ['ingénieur', 'ingenieur', 'senior', 'manager', 'responsable', 'expert'])
            if is_job_senior and is_cv_licence_only:
                continue

            # Calcul du score final
            job_tech = set(w for w in job_text.split() if w in TECHNICAL_SKILLS_LIST)
            common   = cv_tech_keywords.intersection(job_tech)
            skill_score = len(common) / max(len(job_tech), 1) if job_tech else 0.0

            if skill_score > 0:
                final = 0.80 * sem + 0.20 * skill_score
            else:
                final = 0.20 * sem

            score = min(88.0, max(0.0, round(final * 100, 1)))

            if score >= 35:
                results.append({
                    "item": job,
                    "score_global": score,
                    "matched_skills": sorted(common)
                })

        return sorted(results, key=lambda x: x['score_global'], reverse=True)[:5]

# Initialisation unique du moteur de recherche pour toute l'application
ENGINE = None

def get_engine():
    """Génère ou récupère l'instance unique du moteur de matching."""
    global ENGINE
    if ENGINE is None:
        cv_p  = r'C:\Users\Chaima Abdelli\Desktop\PFE CODE\data\cvs_tunisiens.json'
        job_p = r'C:\Users\Chaima Abdelli\Desktop\PFE CODE\data\job_descriptions.csv'
        ENGINE = MatchingEngine(cv_p, job_p)
    return ENGINE
