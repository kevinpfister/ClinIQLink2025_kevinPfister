import os
import json
import re
from datetime import datetime

def enrich_json_file(file_path, additional_info=None):
    """
    Fonction pour enrichir un fichier JSON (benchmark ou raw results)
    
    Args:
        file_path (str): Chemin du fichier JSON
        additional_info (dict, optional): Informations supplémentaires à ajouter
    
    Returns:
        dict: Données enrichies ou None en cas d'erreur
    """
    if additional_info is None:
        additional_info = {}
    
    try:
        # Lire le fichier JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraire les informations du nom de fichier et du chemin
        file_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        file_info = extract_info_from_path_and_name(file_name, dir_name)
        
        # Déterminer le type de fichier
        is_raw_results = 'raw_results' in file_name
        
        # Créer ou mettre à jour les informations du modèle
        if 'model_info' not in data:
            data['model_info'] = {}
        
        # Ajouter les informations extraites du nom de fichier
        data['model_info'].update(file_info)
        
        # Ajouter des informations supplémentaires fournies
        data['model_info'].update(additional_info)
        
        # Ajouter un timestamp s'il n'existe pas déjà
        if 'timestamp' not in data['model_info']:
            data['model_info']['timestamp'] = get_current_timestamp()
        
        # Ajouter les métriques globales pour les fichiers de benchmark
        if not is_raw_results:
            global_metrics = calculate_global_metrics(data)
            data['global_metrics'] = global_metrics
        
        # Écrire les données mises à jour dans le fichier
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Fichier enrichi avec succès: {file_path}")
        return data
    
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return None


def extract_info_from_path_and_name(file_name, dir_path):
    """
    Extraire des informations à partir du nom du fichier et du chemin
    
    Args:
        file_name (str): Nom du fichier
        dir_path (str): Chemin du répertoire
    
    Returns:
        dict: Informations extraites
    """
    info = {}
    
    # Extraire le nombre de questions (pattern: XQA/XQQA où X est un nombre)
    qa_match = re.search(r'(\d+)Q[QA]', file_name) or re.search(r'(\d+)Q[QA]', dir_path)
    if qa_match:
        info['num_questions'] = int(qa_match.group(1))
    
    # Extraire le type de test (Baseline ou RAG)
    if file_name.startswith('Baseline') or '/Baseline' in dir_path:
        info['test_type'] = 'Baseline'
        info['rag_enabled'] = False
    elif file_name.startswith('rag_results') or 'rag_results' in dir_path:
        info['test_type'] = 'RAG'
        info['rag_enabled'] = True
        
        # Extraire le nombre de documents récupérés à partir du chemin
        # Format: rag_results_BM25_30QA_3_MedCorp_Gemini où 3 est le nombre de documents
        docs_match = re.search(r'_(\d+)_(?:MedCorp|SelfCorp|All)', file_name) or \
                     re.search(r'_(\d+)_(?:MedCorp|SelfCorp|All)', dir_path)
        if docs_match:
            info['documents_retrieved'] = int(docs_match.group(1))
    
    # Extraire le nom du modèle
    if 'gemini' in file_name.lower() or 'gemini' in dir_path.lower():
        info['name'] = 'Gemini-2.0-flash' if '2.0' in file_name else 'Gemini'
        info['type'] = 'gemini'
    elif 'Gemma3' in file_name or 'Gemma3' in dir_path:
        info['name'] = 'Gemma-3'
        info['type'] = 'gemma'
    elif 'claude' in file_name.lower() or 'claude' in dir_path.lower():
        info['name'] = 'Claude'
        info['type'] = 'claude'
    elif 'openai' in file_name.lower() or 'openai' in dir_path.lower():
        info['name'] = 'OpenAI'
        info['type'] = 'openai'
    elif 'gpt' in file_name.lower() or 'gpt' in dir_path.lower():
        gpt_match = re.search(r'(gpt-\d+(?:-\w+)?)', file_name, re.IGNORECASE) or \
                    re.search(r'(gpt-\d+(?:-\w+)?)', dir_path, re.IGNORECASE)
        info['name'] = gpt_match.group(1).upper() if gpt_match else 'GPT'
        info['type'] = 'openai'
    
    # Extraire des informations sur le RAG si présent
    rag_docs_match = re.search(r'(\d+)docs', file_name, re.IGNORECASE) or \
                     re.search(r'(\d+)docs', dir_path, re.IGNORECASE)
    if info.get('rag_enabled', False) and rag_docs_match:
        info['num_documents'] = int(rag_docs_match.group(1))
    
    # Extraire le corpus
    corpus_patterns = {
        'MedCorp': r'MedCorp',
        'SelfCorp': r'SelfCorp',
        'All': r'_All'
    }
    
    for corpus, pattern in corpus_patterns.items():
        if re.search(pattern, file_name, re.IGNORECASE) or re.search(pattern, dir_path, re.IGNORECASE):
            info['corpus'] = corpus
            break
    
    # Extraire le type de fichier
    if 'raw_results' in file_name:
        info['file_type'] = 'raw_results'
    elif 'benchmark' in file_name or 'scores' in file_name:
        info['file_type'] = 'benchmark_scores'
    
    return info


def calculate_global_metrics(data):
    """
    Calculer les métriques globales en agrégeant toutes les catégories
    
    Args:
        data (dict): Données du benchmark
    
    Returns:
        dict: Métriques globales
    """
    metrics = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'total_questions': 0
    }
    
    # Catégories à agréger
    categories = ['true_false', 'multiple_choice', 'list']
    
    for category in categories:
        if category in data and isinstance(data[category], dict) and not isinstance(data[category], list):
            metrics['TP'] += data[category].get('TP', 0)
            metrics['FP'] += data[category].get('FP', 0)
            metrics['FN'] += data[category].get('FN', 0)
            metrics['total_questions'] += data[category].get('TP', 0) + data[category].get('FN', 0)
    
    # Calculer les métriques dérivées
    precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
    recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics['accuracy'] = metrics['TP'] / metrics['total_questions'] if metrics['total_questions'] > 0 else 0
    
    return metrics


def get_current_timestamp():
    """
    Obtenir un timestamp au format YYYYMMDD_HHMMSS
    
    Returns:
        str: Timestamp formaté
    """
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S')


def process_directory(dir_path, additional_info=None, recursive=True):
    """
    Traiter tous les fichiers JSON dans un répertoire, y compris les sous-répertoires
    
    Args:
        dir_path (str): Chemin du répertoire contenant les fichiers JSON
        additional_info (dict, optional): Informations supplémentaires à ajouter
        recursive (bool): Parcourir récursivement les sous-répertoires
    """
    if additional_info is None:
        additional_info = {}
    
    try:
        for item in os.listdir(dir_path):
            # Ignorer le dossier "questions" et le fichier config.json
            if item == "questions" or item == "config.json":
                continue
                
            item_path = os.path.join(dir_path, item)
            
            if os.path.isdir(item_path) and recursive:
                # Traiter les sous-répertoires récursivement
                process_directory(item_path, additional_info, recursive)
            elif item.endswith('.json'):
                # Traiter les fichiers JSON (sauf config.json qui a déjà été filtré)
                enrich_json_file(item_path, additional_info)
        
        print(f"Traitement terminé pour le répertoire: {dir_path}")
    
    except Exception as e:
        print(f"Erreur lors du traitement du répertoire {dir_path}: {e}")


# Exemple d'utilisation
if __name__ == '__main__':
    # Exemple de chemin de répertoire (à ajuster selon votre configuration)
    benchmark_dir = './saved_logs/MedCPT'
    
    # Informations supplémentaires à ajouter (optionnel)
    additional_info = {
        # Exemple d'informations additionnelles
        # 'model_version': '1.0.2',
        # 'team': 'NLP Research',
        # 'experiment_id': 'exp_2025_05_18'
    }
    
    # Exemples d'utilisation
    
    # Pour traiter un fichier spécifique:
    # enrich_json_file('./saved_logs/Baseline_30QA_Gemma3_benchmark.json')
    
    # Pour traiter tous les fichiers d'un répertoire (et sous-répertoires):
    process_directory(benchmark_dir, additional_info)
    
    