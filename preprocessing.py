import re
import string
import fitz  # PyMuPDF

# Liste des compétences techniques reconnues par le système
TECHNICAL_SKILLS_LIST = [
    "python", "flask", "django", "java", "spring", "react", "node", "javascript",
    "cpp", "csharp", "dotnet", "stm32", "arm", "embedded", "embarque", "do178",
    "avionique", "trace32", "aerospatial", "rtos", "linux", "autosar", "can",
    "firmware", "sql", "mysql", "postgresql", "mongodb", "docker", "kubernetes",
    "git", "aws", "azure", "ml", "machinelearning", "datascience", "scikit",
    "pandas", "numpy", "devops", "kubernetes", "seo", "marketing", "finance",
    "audit", "excel", "sap", "comptabilite", "pentest", "parefeu", "vulnerabilites",
    "siem", "kalilinux", "forensics", "bilan", "communication", "reseauxsociaux",
    "marketingdigital", "ids", "ips", "soc", "cryptographie", "ethicalhacking"
]

# Liste des mots vides (mots de liaison sans valeur sémantique)
STOPWORDS = set([
    "le","la","les","de","du","des","un","une","au","aux","en","et","ou","par",
    "sur","sous","dans","pour","avec","sans","est","sont","sera","ont","été",
    "ce","ces","cet","cette","mes","tes","ses","nos","vos","leur","leurs",
    "qui","que","quoi","dont","où","comme","mais","donc","car","ni","si",
    "the","and","with","for","from","that","this","these","those","have","been",
    "in","on","at","to","of","by","is","are","was","were","be","an","as","if",
    "or","so","but","not","all","can","will","had","has","its","our","your",
    "développé","travaillé","réalisé","assuré","participé","maintenu","créé",
    "optimisé","élaboré","géré","conduit","piloté","suivi","effectué","mené",
    "conçu","analysé","implémenté","intégré","déployé","configuré","installé",
    "developed","managed","implemented","designed","created","optimized",
    "handled","maintained","supported","led","built","deployed","configured",
    "responsible","worked","done","using","used","working","building","creating",
    "expérience","experience","projet","project","équipe","team","compétence",
    "skill","connaissance","knowledge","mission","tâche","task","maitrise",
    "gestion","conception","mise","oeuvre","pilotage","support","technique",
    "solution","développement","analyse","formation","diplôme","université",
    "ans","an","chez","habite","ville"
])

def protect_special_skills(text):
    """Normalise les termes techniques contenant des caractères spéciaux."""
    # Normalisations demandées par l'utilisateur
    text = re.sub(r"ci-cd", " cicd ", text, flags=re.IGNORECASE)
    text = re.sub(r"ci/cd", " cicd ", text, flags=re.IGNORECASE)
    text = re.sub(r"react\.js", " react ", text, flags=re.IGNORECASE)
    text = re.sub(r"node\.js", " node ", text, flags=re.IGNORECASE)
    text = re.sub(r"vue\.js", " vue ", text, flags=re.IGNORECASE)
    text = re.sub(r"reactjs", " react ", text, flags=re.IGNORECASE)
    text = re.sub(r"nodejs", " node ", text, flags=re.IGNORECASE)
    text = re.sub(r"spring\s+boot", " springboot ", text, flags=re.IGNORECASE)

    # Autres normalisations existantes
    text = re.sub(r"do-178[bc]?", " do178 ", text, flags=re.IGNORECASE)
    text = re.sub(r"do\s+178[bc]?", " do178 ", text, flags=re.IGNORECASE)
    text = re.sub(r"c\+\+", " cpp ", text, flags=re.IGNORECASE)
    text = re.sub(r"c#", " csharp ", text, flags=re.IGNORECASE)
    text = re.sub(r"\.net\b", " dotnet ", text, flags=re.IGNORECASE)
    text = re.sub(r"angular\.js", " angularjs ", text, flags=re.IGNORECASE)
    text = re.sub(r"machine\s+learning", " machinelearning ", text, flags=re.IGNORECASE)
    text = re.sub(r"deep\s+learning", " deeplearning ", text, flags=re.IGNORECASE)
    text = re.sub(r"data\s+science", " datascience ", text, flags=re.IGNORECASE)
    text = re.sub(r"kali\s+linux", " kalilinux ", text, flags=re.IGNORECASE)
    text = re.sub(r"reseaux\s+sociaux", " reseauxsociaux ", text, flags=re.IGNORECASE)
    text = re.sub(r"marketing\s+digital", " marketingdigital ", text, flags=re.IGNORECASE)
    text = re.sub(r"pare-feu", " parefeu ", text, flags=re.IGNORECASE)
    text = re.sub(r"ethical\s+hacking", " ethicalhacking ", text, flags=re.IGNORECASE)
    return text

def detect_language_deterministic(text):
    """Détermine si le texte est en français ou en anglais en comptant les mots clés."""
    text_lower = text.lower()
    fr_markers = len(re.findall(r'\b(le|la|les|de|du|des|et|en|un|une|est|sont|avec|pour|dans|sur)\b', text_lower))
    en_markers = len(re.findall(r'\b(the|and|with|for|in|is|are|of|to|at|by|an|or)\b', text_lower))
    return "fr" if fr_markers >= en_markers else "en"

def clean_text(text):
    """Nettoie le texte en supprimant la ponctuation, les liens et les emails."""
    text = protect_special_skills(text)
    
    # Remplacement des séparateurs spécifiques par des espaces
    text = text.replace('/', ' ')
    text = text.replace('·', ' ')
    text = text.replace('|', ' ')
    text = text.replace(',', ' ')
    
    # Remplacer les tirets entre deux mots par un espace
    text = re.sub(r'(\w)-(\w)', r'\1 \2', text)
    
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\+?216\s*\d[\d\s]{7,}', ' ', text)
    for ch in string.punctuation:
        text = text.replace(ch, ' ')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_filter(text):
    """Découpe le texte en mots et filtre les termes inutiles."""
    tokens = text.split()
    filtered = [
        t for t in tokens
        if t not in STOPWORDS and (len(t) > 1 or t in TECHNICAL_SKILLS_LIST)
    ]
    return filtered

def get_clean_pipeline(text):
    """Exécute toute la chaîne de traitement sur un texte brut."""
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned)
    return " ".join(tokens)

def build_weighted_text(cv_data):
    """Construit un texte enrichi pour le matching en favorisant les compétences et l'expérience."""
    skills = " ".join(cv_data.get('competences', []))
    exp    = cv_data.get('experience', "")
    edu    = cv_data.get('diplome', "") + " " + cv_data.get('universite', "")
    return f"{skills} {skills} {exp} {exp} {edu}"

def extract_text_from_pdf(pdf_path):
    """Extrait le contenu textuel d'un fichier PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + " "
    except Exception as e:
        print(f"Erreur lecture PDF: {e}")
    return text
