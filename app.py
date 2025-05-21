import os
import logging
import pandas as pd
import time
import gc
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import pickle
from pathlib import Path

# Download NLTK data at startup
nltk.download('stopwords')
nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assume NLTK resources are now available
stop_words = set(stopwords.words('english')).union({"workbooks", "reports", "dashboards"})

# Toggle for development vs full run
DEV_MODE = True
MAX_TERMS = 3000 if DEV_MODE else None
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Paths
info_xlsx_path = "info.xlsx"  # Updated to relative path in the root directory
cache_file = CACHE_DIR / "app_data.pkl"

# Token configuration for Tableau REST API
token_name = "TableauSearchAgentOG"
personal_access_token = "z91zAqhMRZOIhtOAXl4roA==:akhkqQ6DZRoLvB5DFta0WVVzPw9L6Ava"  # Replace with your actual Token Secret
site = 'axosfinancialproduction'
server_url = 'https://prod-useast-b.online.tableau.com'
TABLEAU_API_VERSION = "3.19"  # Adjust to the latest version

# Global variables for authentication token and site ID
auth_token = None
site_id = None

dynamic_concept_dict = {}
all_concept_terms = []
tableau_tfidf_matrix = None
tableau_features = None
tableau_df = None
vectorizer = None
expand_cache = {}  # Cache for query expansion

def sign_in_to_tableau():
    global auth_token, site_id
    if not personal_access_token:
        logger.warning("personal_access_token not set; cannot sign in to Tableau API.")
        return False

    url = f"{server_url}/api/{TABLEAU_API_VERSION}/auth/signin"
    payload = {
        "credentials": {
            "personalAccessTokenName": token_name,
            "personalAccessTokenSecret": personal_access_token,
            "site": {"contentUrl": site}
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Sign-in response: {data}")
        credentials = data.get("credentials", {})
        auth_token = credentials.get("token")
        site_id = credentials.get("site", {}).get("id")
        if not auth_token or not site_id:
            logger.error("Failed to retrieve auth token or site ID from sign-in response.")
            return False
        logger.info(f"Successfully signed in to Tableau. Auth Token: {auth_token}, Site ID: {site_id}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error signing in to Tableau API: {str(e)}, Response: {getattr(e.response, 'text', 'No response')}")
        return False

# Load or build cached data
def load_or_build_data():
    global dynamic_concept_dict, all_concept_terms, tableau_tfidf_matrix, tableau_features, tableau_df, vectorizer

    # Check if Parquet files exist; if not, convert from Excel
    sheets = {
        'tableau': ['name', 'id', 'path', 'url'],
        'xml': ['file_name', 'path', 'url', 'extracted_files'],
        'sql_table': ['table_name', 'file_name', 'url'],
        'sql_field': ['table_name', 'file_name', 'url', 'table_alias'],
        'sql_table_raw': ['table_name', 'file_name', 'url', 'table_alias'],
        'sql_table_count': ['table_name'],
        'sql_field_raw': ['field_name', 'table_name', 'table_alias'],
        'sql_field_count': ['field_name']
    }
    parquet_files = {sheet: CACHE_DIR / f"{sheet}.parquet" for sheet in sheets}
    all_parquet_exist = all([f.exists() for f in parquet_files.values()])

    if not all_parquet_exist:
        logger.info("Converting Excel to Parquet for faster loading...")
        for sheet, cols in sheets.items():
            df = pd.read_excel(info_xlsx_path, sheet_name=sheet, usecols=cols, engine='openpyxl')
            df.to_parquet(parquet_files[sheet])
        logger.info("Excel to Parquet conversion completed.")

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 5:
                    logger.warning("Old cache format detected; rebuilding cache.")
                    dynamic_concept_dict, all_concept_terms, tableau_tfidf_matrix, tableau_features, tableau_df = data
                    vectorizer = None
                elif len(data) == 6:
                    dynamic_concept_dict, all_concept_terms, tableau_tfidf_matrix, tableau_features, tableau_df, vectorizer = data
                    logger.info("Loaded cached data successfully.")
                    return
                else:
                    logger.error("Invalid cache format; rebuilding cache.")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}; rebuilding cache.")

    # Load data from Parquet
    start_time = time.time()
    try:
        dataframes = {sheet: pd.read_parquet(parquet_files[sheet]) for sheet in sheets}
        tableau_df = dataframes['tableau']
        xml_df = dataframes['xml']
        sql_table_df = dataframes['sql_table']
        sql_field_df = dataframes['sql_field']
        sql_table_raw_df = dataframes['sql_table_raw']
        sql_table_count_df = dataframes['sql_table_count']
        sql_field_raw_df = dataframes['sql_field_raw']
        sql_field_count_df = dataframes['sql_field_count']
        logger.info(f"All sheets loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error loading Parquet files: {str(e)}")
        return

    # Free memory
    del dataframes
    gc.collect()

    # Build dynamic concept dictionary
    start_time = time.time()
    data_sources = [
        (tableau_df, ['name']),
        (sql_table_df, ['table_name']),
        (sql_field_df, ['table_name']),
        (sql_table_raw_df, ['table_name']),
        (xml_df, ['file_name'])
    ]
    for df, cols in data_sources:
        if not df.empty:
            for col in cols:
                if col in df.columns:
                    terms = df[col].dropna().astype(str).str.lower().apply(lambda x: word_tokenize(x)).sum()
                    terms = [t for t in terms if t.isalnum() and t.lower() not in stop_words]
                    all_concept_terms.extend(terms)
                    for term in terms:
                        if term not in dynamic_concept_dict:
                            dynamic_concept_dict[term] = set()
                        dynamic_concept_dict[term].add(term)
    all_concept_terms = list(set(all_concept_terms))
    if DEV_MODE and len(all_concept_terms) > MAX_TERMS:
        all_concept_terms = all_concept_terms[:MAX_TERMS]
    logger.info(f"Built dynamic concept dictionary with {len(dynamic_concept_dict)} entries in {time.time() - start_time:.2f} seconds.")

    # Precompute TF-IDF for tableau_df
    start_time = time.time()
    if not tableau_df.empty:
        try:
            tableau_df['searchable_text'] = tableau_df.apply(
                lambda row: f"{row.get('name', '')}".lower(),
                axis=1
            )
            tableau_df = tableau_df[['searchable_text', 'url', 'name', 'path', 'id']]  # Include id for API lookup
            vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=2000, min_df=2, max_df=0.95)
            tableau_tfidf_matrix = vectorizer.fit_transform(tableau_df['searchable_text'])
            tableau_features = vectorizer.get_feature_names_out()
            logger.info(f"Precomputed TF-IDF matrix in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error precomputing TF-IDF: {str(e)}")

    # Cache data including vectorizer
    with open(cache_file, 'wb') as f:
        pickle.dump((dynamic_concept_dict, all_concept_terms, tableau_tfidf_matrix, tableau_features, tableau_df, vectorizer), f)
    logger.info("Cached data to disk.")

# Sign in to Tableau on startup
if not sign_in_to_tableau():
    logger.error("Failed to initialize Tableau API authentication. Metadata fetching will not work.")

# Flask app with template folder set to root directory
app = Flask(__name__, template_folder='.')
CORS(app, origins=['https://ogirbau.github.io'])

def expand_query(user_query):
    if user_query in expand_cache:
        logger.info(f"Using cached expansion for query: {user_query}")
        return expand_cache[user_query]

    logger.info(f"Expanding user query: {user_query}")
    start = time.time()

    user_tokens = [token.lower() for token in word_tokenize(user_query) if token.isalnum() and token.lower() not in stop_words]
    if not user_tokens:
        return user_tokens

    limited_terms = all_concept_terms if not DEV_MODE else all_concept_terms[:MAX_TERMS]
    corpus = list(set(user_tokens + limited_terms))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = tfidf_matrix[0:len(user_tokens)]
    concept_vecs = tfidf_matrix[len(user_tokens):]
    similarity_scores = cosine_similarity(query_vec, concept_vecs)

    threshold = 0.3
    top_matches = np.argwhere(similarity_scores > threshold)
    expanded_terms = set(user_tokens)

    for _, j in top_matches:
        if j < len(concept_vecs.toarray()):
            expanded_terms.add(corpus[len(user_tokens) + j])

    end = time.time()
    logger.info(f"Expanded query terms: {expanded_terms}")
    logger.info(f"Query expansion took {end - start:.2f} seconds.")
    
    expand_cache[user_query] = list(expanded_terms)
    return list(expanded_terms)

def fetch_tableau_metadata(workbook_id):
    if not auth_token or not site_id:
        logger.warning("No auth token or site ID available; skipping API call.")
        return {"last_modified": "N/A", "owner": "N/A", "description": "N/A"}

    if not workbook_id or not isinstance(workbook_id, str):
        logger.warning(f"Invalid workbook_id: {workbook_id}; skipping API call.")
        return {"last_modified": "N/A", "owner": "N/A", "description": "N/A"}

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Tableau-Auth": auth_token
    }
    url = f"{server_url}/api/{TABLEAU_API_VERSION}/sites/{site_id}/workbooks/{workbook_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info(f"API response for workbook {workbook_id}: {data}")
        workbook = data.get("workbook", {})
        last_modified = workbook.get("updatedAt", "N/A")
        owner = workbook.get("owner", {}).get("name", "N/A") if workbook.get("owner") else "N/A"
        description = workbook.get("description", "N/A")
        logger.info(f"Extracted metadata - Last Modified: {last_modified}, Owner: {owner}, Description: {description}")
        return {
            "last_modified": last_modified,
            "owner": owner,
            "description": description
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching metadata for workbook {workbook_id}: {str(e)}, Response: {getattr(e.response, 'text', 'No response')}")
        # If token expired (401), try to re-authenticate
        if getattr(e.response, 'status_code', 0) == 401:
            logger.info("Possible token expiration. Attempting to re-authenticate...")
            if sign_in_to_tableau():
                headers["X-Tableau-Auth"] = auth_token
                try:
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    logger.info(f"API response after re-auth for workbook {workbook_id}: {data}")
                    workbook = data.get("workbook", {})
                    last_modified = workbook.get("updatedAt", "N/A")
                    owner = workbook.get("owner", {}).get("name", "N/A") if workbook.get("owner") else "N/A"
                    description = workbook.get("description", "N/A")
                    logger.info(f"Extracted metadata after re-auth - Last Modified: {last_modified}, Owner: {owner}, Description: {description}")
                    return {
                        "last_modified": last_modified,
                        "owner": owner,
                        "description": description
                    }
                except requests.exceptions.RequestException as e2:
                    logger.error(f"Error fetching metadata after re-auth for workbook {workbook_id}: {str(e2)}, Response: {getattr(e2.response, 'text', 'No response')}")
        return {"last_modified": "N/A", "owner": "N/A", "description": "N/A"}

def search_workbooks(query, limit=None):
    start = time.time()
    if tableau_df.empty or tableau_tfidf_matrix is None or vectorizer is None:
        logger.warning("tableau_df is empty, TF-IDF not computed, or vectorizer missing; returning empty results.")
        return []

    expanded_terms = expand_query(query)
    expanded_query = ' '.join(expanded_terms)
    query_vec = vectorizer.transform([expanded_query])
    similarity_scores = cosine_similarity(query_vec, tableau_tfidf_matrix)

    scores = similarity_scores[0]
    top_indices = scores.argsort()[::-1]
    is_acronym = any(word in query.lower() for word in ["hmda", "ulp"])
    
    logger.info(f"Top similarity scores: {scores[top_indices[:5]]}")
    
    default_limit = 20  # Target 20 dashboards
    max_results = limit if limit is not None else default_limit

    results = []
    for i in top_indices:
        if scores[i] > 0.1 and len(results) < max_results:
            row = tableau_df.iloc[i]
            # Filter to include only dashboard URLs (e.g., containing /workbooks/)
            if 'url' in row and isinstance(row['url'], str) and '/workbooks/' in row['url'].lower():
                workbook_id = row.get('id', '')
                logger.info(f"Fetching metadata for workbook_id: {workbook_id}")
                metadata = fetch_tableau_metadata(workbook_id)
                results.append({
                    "name": row.get('name', 'Unnamed Dashboard'),  # From info.xlsx
                    "webpage_url": row.get('url', 'N/A'),  # From info.xlsx
                    "path": row.get('path', ''),  # From info.xlsx
                    "last_modified": metadata.get("last_modified", "N/A"),  # From REST API
                    "owner": metadata.get("owner", "N/A"),  # From REST API
                    "description": metadata.get("description", "N/A")  # From REST API
                })

    # If fewer than 20 dashboards, include all valid results up to the limit
    if len(results) < max_results and len(results) < len(top_indices):
        for i in top_indices[len(results):]:
            if scores[i] > 0.1 and len(results) < max_results:
                row = tableau_df.iloc[i]
                if 'url' in row and isinstance(row['url'], str) and '/workbooks/' in row['url'].lower():
                    workbook_id = row.get('id', '')
                    logger.info(f"Fetching metadata for workbook_id: {workbook_id}")
                    metadata = fetch_tableau_metadata(workbook_id)
                    results.append({
                        "name": row.get('name', 'Unnamed Dashboard'),  # From info.xlsx
                        "webpage_url": row.get('url', 'N/A'),  # From info.xlsx
                        "path": row.get('path', ''),  # From info.xlsx
                        "last_modified": metadata.get("last_modified", "N/A"),  # From REST API
                        "owner": metadata.get("owner", "N/A"),  # From REST API
                        "description": metadata.get("description", "N/A")  # From REST API
                    })

    end = time.time()
    logger.info(f"Search completed in {end - start:.2f} seconds, returning {len(results)} results.")
    return results

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return jsonify({"error": "Template rendering failed", "details": str(e)}), 500

@app.route('/expand', methods=['POST'])
def expand():
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    expanded = expand_query(user_query)
    return jsonify({'expanded_query': expanded})

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    limit = request.args.get('limit', type=int)
    if not query:
        logger.warning("No query parameter provided.")
        return jsonify({"error": "No query parameter provided"}), 400
    logger.info(f"Received search query: {query}")
    results = search_workbooks(query, limit)
    logger.info(f"Returning {len(results)} results for query: {query}")
    return jsonify(results)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    # Load data on startup
    load_or_build_data()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False, threaded=True)
