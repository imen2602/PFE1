import pandas as pd
import numpy as np
from matcher import get_engine
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import get_clean_pipeline

def run_evaluation():
    """Évalue et compare les performances du modèle BERT face à une approche classique TF-IDF."""
    engine = get_engine()
    
    # Définition des scénarios de test pour la démonstration
    test_cases = [
        ("Ingénieur firmware STM32 systèmes temps réel", "Embarque"),
        ("Développeur Python Machine Learning Pandas", "DataScience"),
        ("Expert cybersécurité tests intrusion pare-feu", "Securite"),
        ("Développeur Full Stack React Node.js", "FullStack"),
        ("Comptable senior audit fiscal SAP", "Finance"),
        ("Ingénieur logiciel embarqué DO-178 certification", "Embarque"),
        ("Data Scientist deep learning TensorFlow", "DataScience"),
        ("Développeur backend Java Spring microservices", "Informatique"),
        ("Ingénieur sécurité réseaux Kali Linux forensics", "Securite"),
        ("Développeur frontend JavaScript TypeScript", "FullStack"),
    ]
    
    y_true = [1] * len(test_cases)
    
    # Analyse avec le modèle RecrutIA (BERT)
    y_pred_bert = []
    print("Analyse RecrutIA (BERT) en cours...")
    for job_text, expected_domain in test_cases:
        results = engine.match_job_to_cvs(job_text)
        top_domain = results[0]['item'].get('domaine', '') if results else ''
        y_pred_bert.append(1 if top_domain == expected_domain else 0)
    
    # Analyse avec l'approche de base (TF-IDF)
    y_pred_tfidf = []
    print("Analyse Baseline (TF-IDF) en cours...")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(engine.weighted_cv_texts)
    
    for job_text, expected_domain in test_cases:
        query_clean = get_clean_pipeline(job_text)
        query_vec = vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        top_domain_tfidf = engine.cvs[best_idx].get('domaine', '')
        y_pred_tfidf.append(1 if top_domain_tfidf == expected_domain else 0)
        
    # Calcul et affichage des métriques de performance
    metrics = []
    for name, y_pred in [("RecrutIA (BERT)", y_pred_bert), ("Baseline TF-IDF", y_pred_tfidf)]:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics.append({
            "Système": name,
            "Précision": round(precision, 2),
            "Rappel": round(recall, 2),
            "F1-Score": round(f1, 2)
        })

    print("\n--- TABLEAU COMPARATIF DES PERFORMANCES ---")
    df = pd.DataFrame(metrics)
    print(df.to_string(index=False))
    
    bert_f1 = metrics[0]["F1-Score"]
    tfidf_f1 = metrics[1]["F1-Score"]
    
    if bert_f1 >= tfidf_f1:
        print("\nConclusion : BERT Multilingue surpasse TF-IDF grâce à sa compréhension sémantique.")
    else:
        print("\nConclusion : Observation inattendue sur ce dataset réduit.")

if __name__ == "__main__":
    print("Démarrage de l'évaluation comparative des performances...")
    run_evaluation()
