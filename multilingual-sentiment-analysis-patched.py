import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
import requests
from datetime import datetime
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import pandas as pd
import traceback
import transformers
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSequenceClassification
transformers.logging.set_verbosity_error()
import langid
from scipy.special import softmax
import numpy as np
import psycopg2
import re
from typing import Tuple, Dict
import torch
import emoji
from langdetect import detect as langdetect_detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import json
import unicodedata
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable.*")
#warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import logging
from transformers import pipeline
from functools import lru_cache

logging.basicConfig(
    filename='sentiment_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable.*")
tf.experimental.numpy.experimental_enable_numpy_behavior()

def connect_to_db():
    try:
        return psycopg2.connect(
            dbname='page_comments',
            user='scrapper_user',
            password='scRaPPer_user',
            host='10.20.10.42',
            port='5432'
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        return None
'''
def update_database(valid_df):
    print(f"Updating {len(valid_df)} records in database...")
    conn = connect_to_db()
    if not conn:
        return
    updates_count = 0
    batch_size = 100
    required_columns = ['id', 'detected_language', 'sentiment_label', 'sentiment_score', 'is_question', 'question_indicators', 'is_spam']
    missing = [col for col in required_columns if col not in valid_df.columns]
    if missing:
        print(f"ERROR: Missing columns for DB update: {missing}")
        return
    try:
        # Prepare update data
        update_data = []
        for _, row in valid_df.iterrows():
            if row['is_spam']:
                # For spam comments
                update_data.append((
                    row['detected_language'],  
                    None,                      # sentiment_label
                    None,                      # sentiment_score
                    None,                      # is_question
                    json.dumps(row.get('question_indicators', {})),  # question_indicators
                    True,                      # is_spam
                    row['id']                  # id
                ))
            else:
                update_data.append((
                    row['detected_language'],
                    row['sentiment_label'] ,
                    #float(row['sentiment_score']),
                    float(row['sentiment_score']) ,
                    #row.get('is_question', False), 
                    bool(row['is_question']),
                    json.dumps(row.get('question_indicators', {})), 
                    #row.get('is_spam', False),
                    bool(row['is_spam']),
                    row['id']
            ))
        with conn:
            with conn.cursor() as cursor:
                for i in range(0, len(update_data), batch_size):
                    batch = update_data[i:i + batch_size]
                    try:
                        cursor.executemany(
                            """UPDATE facebook_comments
                               SET detected_language = %s,
                                   sentiment_label = %s,
                                   sentiment_score = %s,
                                   is_question = %s,
                                   question_indicators = %s,
                                   is_spam = %s
                               WHERE id = %s""",
                            batch
                        )
                        updates_count += len(batch)
                        #updates_count += cursor.rowcount
                        print(f"Updated {updates_count}/{len(valid_df)} records...")
                    except Exception as e:
                        print(f"Error updating batch starting at record {i}: {e}")
                        print(f"Problematic batch data: {batch}")
                        conn.rollback()
                        raise
        print(f"Successfully updated {updates_count} records")
    except Exception as e:
        print(f"Error in database update process: {e}")
    finally:
        conn.close()

'''

def update_database(valid_df):
    print(f"Updating {len(valid_df)} records in database...")
    conn = connect_to_db()
    if not conn:
        return
    updates_count = 0
    batch_size = 100
    required_columns = ['id', 'detected_language', 'sentiment_label', 'sentiment_score', 'is_question', 'question_indicators', 'is_spam']
    missing = [col for col in required_columns if col not in valid_df.columns]
    if missing:
        print(f"ERROR: Missing columns for DB update: {missing}")
        return
    
    try:
        # Prepare update data with proper null handling
        update_data = []
        for _, row in valid_df.iterrows():
            # Handle None/nan values
            detected_lang = row['detected_language'] if pd.notna(row['detected_language']) else None
            sentiment_label = row['sentiment_label'] if pd.notna(row['sentiment_label']) else None
            sentiment_score = float(row['sentiment_score']) if pd.notna(row['sentiment_score']) else None
            is_question = bool(row['is_question']) if pd.notna(row['is_question']) else None
            
            # Handle question_indicators - convert 'NaN' to None and ensure valid JSON
            question_indicators = row.get('question_indicators', {})
            if isinstance(question_indicators, str) and question_indicators.strip() == 'NaN':
                question_indicators = None
            elif isinstance(question_indicators, (dict, list)):
                question_indicators = json.dumps(question_indicators)
            elif question_indicators is None:
                pass  # Keep as None
            else:
                # Invalid format, default to empty dict
                question_indicators = json.dumps({})
                
            is_spam = bool(row['is_spam']) if pd.notna(row['is_spam']) else None
            
            update_data.append((
                detected_lang,
                sentiment_label,
                sentiment_score,
                is_question,
                question_indicators,
                is_spam,
                row['id']
            ))

        with conn:
            with conn.cursor() as cursor:
                for i in range(0, len(update_data), batch_size):
                    batch = update_data[i:i + batch_size]
                    try:
                        cursor.executemany(
                            """UPDATE facebook_comments
                               SET detected_language = %s,
                                   sentiment_label = %s,
                                   sentiment_score = %s,
                                   is_question = %s,
                                   question_indicators = %s,
                                   is_spam = %s
                               WHERE id = %s""",
                            batch
                        )
                        updates_count += len(batch)
                        print(f"Updated {updates_count}/{len(valid_df)} records...")
                    except Exception as e:
                        print(f"Error updating batch starting at record {i}: {e}")
                        print(f"Problematic batch data: {batch}")
                        conn.rollback()
                        raise
                        
        print(f"Successfully updated {updates_count} records")
    except Exception as e:
        print(f"Error in database update process: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def check_database_connection():    
    import psycopg2
    conn = psycopg2.connect(dbname='page_comments',
            user='scrapper_user',
            password='scRaPPer_user',
            host='10.20.10.42',
            port='5432')
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'facebook_comments'
        );
    """)
    table_exists = cur.fetchone()[0]
    print(f"Table exists: {table_exists}")
    
    # Check row count
    cur.execute("SELECT COUNT(*) FROM facebook_comments;")
    count = cur.fetchone()[0]
    print(f"Total comments in DB: {count}")
    
    conn.close()

check_database_connection()
def extract_comments():
    conn = connect_to_db()
    if not conn:
        return None
    
    #query = "SELECT id, post_id, commenter_name, comment_time, comment_message FROM facebook_comments;"
    query = """
            SELECT id, post_id, commenter_name, comment_time, comment_message 
            FROM facebook_comments  
            WHERE (detected_language IS NULL OR sentiment_label IS NULL )
            AND comment_message IS NOT NULL 
            AND comment_message != ''
            AND LENGTH(TRIM(comment_message)) > 0 
            ORDER BY comment_time DESC
            LIMIT 1000
        """
    try:
        df = pd.read_sql_query(query, conn)
        # Remove null, empty, or whitespace-only comments
        df = df[df['comment_message'].notna()]                       # Drop NaNs
        df = df[df['comment_message'].str.strip().astype(bool)]     # Drop empty/whitespace

        return df
    except Exception as e:
        print(f"Data extraction error: {e}")
        return None
    finally:
        conn.close()
import json

def extract_commenter_name(raw_json):
    try:
        if isinstance(raw_json, str):
            raw_data = json.loads(raw_json)
        elif isinstance(raw_json, dict):
            raw_data = raw_json
        else:
            return 'Unknown'

        if 'from' in raw_data and 'name' in raw_data['from']:
            return raw_data['from']['name']
        else:
            return 'Anonymous'
    except Exception as e:
        return 'Unknown'

def is_valid_comment(text):

    if not isinstance(text, str) or not text.strip():
        return False
    
    
    # emoji-only comments  Kawther verifi is_emoji ida tekhdem correctly
    if all(is_emoji(c) for c in text if c.strip()):
        return True  # Consider emoji-only comments as valid
    
    # minimum meaningful content
    cleaned = clean_text(text)
    return len(cleaned) >= 2  # At least 2 characters after cleaning

def process_comments(df):
    # Mark spam first
    df['is_spam'] = df['comment_message'].apply(is_spam_comment)
    
    # Clean all comments (including spam for debugging)
    df['cleaned_message'] = df['comment_message'].apply(clean_text)
    
    # Validate (spam comments automatically invalid)
    df['is_valid'] = ~df['is_spam'] & df['comment_message'].apply(is_valid_comment)
    
    # Separate streams
    spam_df = df[df['is_spam']]
    valid_df = df[df['is_valid']]
    invalid_df = df[~df['is_spam'] & ~df['is_valid']]
    
    print(f"Classification Results:")
    print(f"- Valid comments: {len(valid_df)}")
    print(f"- Spam comments: {len(spam_df)}")
    print(f"- Invalid (empty/broken): {len(invalid_df)}")
    
    return valid_df

def clean_text(text):
    
    if is_spam_comment(text):
        return {'label': None , 'score': None , 'spam': True}
    '''
    if pd.isna(text) or not isinstance(text, str):
        print(f"Invalid text: {text}")
        return ""
    '''
    if pd.isna(text) or not isinstance(text, str):
        logging.debug(f"[clean_text] Invalid input: {text}")
        return ""
    
    # Preserve emojis
    emoji_chars = [c for c in text if is_emoji(c)]
    logging.debug(f"[clean_text] Original: {text}, Emojis: {emoji_chars}")
    original = text
    # Normalize Arabic characters
    text = re.sub(r'[إأٱآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[ؤئ]', 'ء', text)

    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F]', '', text)
    
    # Standard cleaning
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text) # u could consider it as a mention remover
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\"'\/\\()\[\]{}\-_+=*&^%$#@<>|~`؟،؛]", "", text)

    # If we stripped everything but had emojis, restore them
    if not text.strip() and emoji_chars:
        cleaned = ' '.join(emoji_chars)
        logging.debug(f"[clean_text] Restored emojis: {cleaned}")
        return cleaned
    
    logging.debug(f"[clean_text] Cleaned: {text}")
    return text.strip()
    
def debug_cleaning(df):
    print("\n=== RAW DATA SAMPLE ===")
    print(df[['id', 'comment_message']].head(3).to_dict('records'))
    
    # Check initial null/empty status
    initial_empty = (df['comment_message'].astype(str).str.strip() == '').sum()
    initial_null = df['comment_message'].isna().sum()
    print(f"\nInitial - Empty: {initial_empty}, Null: {initial_null}, Total: {len(df)}")
    
    # Apply cleaning step-by-step
    df['step1_str'] = df['comment_message'].astype(str)
    df['step2_cleaned'] = df['step1_str'].apply(clean_text)
    df['step3_lower'] = df['step2_cleaned'].str.lower()
    df['step4_stripped'] = df['step3_lower'].str.strip()
    
    print("\n=== CLEANING STEPS ===")
    sample_idx = df.index[0] if len(df) > 0 else None
    if sample_idx:
        print("Single comment transformation:")
        print(f"Original: {df.loc[sample_idx, 'comment_message']}")
        print(f"Step1 (to_str): {df.loc[sample_idx, 'step1_str']}")
        print(f"Step2 (cleaned): {df.loc[sample_idx, 'step2_cleaned']}")
        print(f"Step3 (lower): {df.loc[sample_idx, 'step3_lower']}")
        print(f"Step4 (stripped): {df.loc[sample_idx, 'step4_stripped']}")
    
    return df
def detect_language(text):

    if pd.isna(text) or text == '[null]' or not isinstance(text, str) or not text.strip():
        return 'unknown'
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return 'unknown'
    
    try:
        lang, conf = langid.classify(cleaned_text)
        
        darija_indicators = [
            'wesh', 'kifesh', 'kayen', 'nchalah', 'khoya', 'sahbi', 'inchallah', 
            'saha', 'mlih', 'rani', 'smahli', 'labas', 'saha', 'zin', 'bghit',
            'ma3lish', 'mlih', 'hamdulillah', 'haniyan', 'douk', 'ghedwa',
            'temma', 'fhad', 'nhar', 'yallah', 'bezaf', 'win', 'kif',
            'wlad', 'bent', 'rajel', 'mra', 'weld', 'lahwa', 'kber', 'sgher',
            'كيفاش', 'واش', 'كاين', 'خويا', 'صاحبي', 'نشالله', 'انشالله', 'زين',
            'غادي', 'بغيت', 'مكاين', 'فهمت', 'شكرا', 'بزاف', 'دروك', 'هدا',
             'درك','غدوا', 'وين', 'كيف', 'ولاد', 'بنت', 'راجل', 'مرا', 'ولد',
            'لاباس', 'نتا', 'حنا', 'شحال', 'بنادم'
        ]
        
        french_darija_mix = any(word in cleaned_text.lower() for word in ['mais', 'donc', 'alors', 'voila']) and \
                           any(word in cleaned_text.lower() for word in darija_indicators)
        
        normalized_text = cleaned_text.lower()
        text_words = normalized_text.split()
        total_words = len(text_words)
        
        darija_matches = [word for word in darija_indicators if word in normalized_text]
        darija_match_ratio = len(darija_matches) / max(1, total_words)
        
        contains_arabic_script = bool(re.search(r'[\u0600-\u06FF]', cleaned_text))
        contains_latin_script = bool(re.search(r'[a-zA-Z]', cleaned_text))
        
        if darija_matches or french_darija_mix:
            if contains_arabic_script and any(word in normalized_text for word in ['واش', 'كيفاش', 'مزيان']):
                return 'darija'
            elif contains_latin_script and any(word in normalized_text for word in ['wesh', 'kifesh', 'labas']):
                return 'darija'
            elif darija_match_ratio > 0.2:
                return 'darija'
        
        #if conf < 0.75 or (lang not in ['ar', 'fr', 'en'] and total_words > 3):
        if conf < 0.6 or (lang not in ['ar', 'fr', 'en'] and total_words > 2):    
            try:
                lang_detect = langdetect_detect(cleaned_text)                
                if lang_detect in ['ar', 'fr', 'en']:
                    if lang_detect != lang:
                        if contains_arabic_script and lang_detect == 'ar':
                            if any(word in normalized_text for word in darija_indicators):
                                return 'darija'
                            return 'ar'
                        elif contains_latin_script and lang_detect in ['fr', 'en']:
                            return lang_detect
            except LangDetectException:
                pass
        
        if lang == 'ar' or contains_arabic_script:
            if any(word in normalized_text for word in darija_indicators):
                return 'darija'
            return 'ar'
        if len(text_words) <= 3 and any(word in normalized_text.split() for word in 
                              ['bon', 'merci', 'oui', 'non', 'courage']):
            return 'fr'
        if lang == 'fr' or (contains_latin_script and any(word in normalized_text.split() for word in 
                                                         ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 
                                                          'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais'])):
            return 'fr'
        
        if lang == 'en' or (contains_latin_script and any(word in normalized_text.split() for word in 
                                                         ['the', 'an', 'of', 'in', 'on', 'at', 'for', 'with', 
                                                          'and', 'but', 'or', 'so', 'because', 'if'])):
            return 'en'
        
        if contains_arabic_script:
            return 'darija'
        
        return lang if lang in ['en', 'fr', 'ar'] else 'unknown'
    
    except Exception as e:
        print(f"Language detection error: {e} for text: {text[:50]}...")
        return 'unknown'

model_cache = {}

def load_model(model_name):
    print(f"Loading model: {model_name}")
    try: 
        if os.path.exists(model_name):
            # Local model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            if "arabic" in model_name.lower() or "camelbert" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def load_sentiment_models():
    return {
        'en': "/home/benslimane_m/scrapper/scrapper_2/NLPTown-Sentiment-Offline/",#work
        'fr': "/home/benslimane_m/scrapper/scrapper_2/NLPTown-Sentiment-Offline/",#work
        #'ar': "UBC-NLP/MARBERT", # More stable
        #'ar': "Abdo36/Arabert-Sentiment-Analysis-ArSAS", 
        #'darija': "alger-ia/dziribert", #work
        #'darija':"Abdo36/Arabert-Sentiment-Analysis-ArSAS"
        #'ar': "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
        #'darija': "/home/benslimane_m/scrapper/scrapper_2/NLPTown-Sentiment-Offline/" 
        'ar': "/home/benslimane_m/scrapper/scrapper_2/Arabert-Sentiment-Offline/",
        'darija': "/home/benslimane_m/scrapper/scrapper_2/Arabert-Sentiment-Offline/"

    }
import csv
import csv

def load_emoji_terms_map():
    emoji_sentiment_map = {}
    label_to_score = {
        'positive': 1.0,
        'neutre': 0.5,
        'neutral': 0.5,
        'negative': 0.0,
        'negatif': 0.0
    }

    try:
        with open('emoji-map.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                emoji = row.get('emoji')
                label = row.get('label', '').strip().lower()
                score = label_to_score.get(label)
                if emoji and score is not None:
                    emoji_sentiment_map[emoji] = score
    except Exception as e:
        logging.error(f"Error loading emoji sentiment map: {e}")
    
    return emoji_sentiment_map  

def load_prayer_terms_map():
    prayer_terms_map = {}
    label_to_score = {
        'positive': 0.3,
        'neutral': 0.0,
        'negative': -0.3
    }

    try:
        with open('prayer-map2.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row.get('term')
                label = row.get('label', '').strip().lower()
                score = label_to_score.get(label)
                if term and score is not None:
                    prayer_terms_map[term] = score
    except Exception as e:
        logging.error(f"Failed to load prayer term map: {e}")

    return prayer_terms_map


def is_emoji(char):

    return char in emoji.UNICODE_EMOJI['en'] if hasattr(emoji, 'UNICODE_EMOJI') else emoji.demojize(char) != char

'''
def is_spam_comment(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    hashtag_count = text.count('#')
    link_count = len(re.findall(r'(https?://\S+|www\.\S+)', text))
    promotional_phrases = [
        'join', 'subscribe', 'follow', 'click here', 'link in bio',
        'اشترك', 'تابع', 'رابط', 'انضم', 'سجل'
    ]
        
    is_promotional = any(phrase in text.lower() for phrase in promotional_phrases)
    return hashtag_count >= 3 or link_count >= 1 or is_promotional
'''
def is_emoji(char):

    return char in emoji.UNICODE_EMOJI['en'] if hasattr(emoji, 'UNICODE_EMOJI') else emoji.demojize(char) != char
'''
def is_emoji_only(text):
    if not isinstance(text, str):
        return False
    #cleaned = text.strip()
    cleaned = ''.join(c for c in text if not unicodedata.category(c).startswith('C')).strip()
    return len(cleaned) > 0 and all(is_emoji(c) for c in cleaned if c.strip())
'''
def is_emoji_only(text):
    if not isinstance(text, str):
        logging.debug(f"[is_emoji_only] Input is not a string: {text}")
        return False
    
    emojis = [c for c in text if c in emoji.UNICODE_EMOJI['en']]
    cleaned = ''.join(emojis).strip()
    is_only_emoji = len(cleaned) > 0 and all(c in emoji.UNICODE_EMOJI['en'] for c in text if c.strip())
    logging.debug(f"[is_emoji_only] Text: {text}, Cleaned: {cleaned}, Is emoji-only: {is_only_emoji}")
    return is_only_emoji

def is_spam_comment(text: str) -> bool:

    if not isinstance(text, str) or not text.strip():
        return False
        # Skip emoji-only comments from spam detection
    if is_emoji_only(text):
        return False
    # Remove emojis from the text for spam detection
    text_without_emojis = ''.join(c for c in text if not is_emoji(c))
    # Spam patterns
    spam_patterns = [
        r'http[s]?://\S+',  # URLs
        r'www\.\S+',         # URLs
        r'\b(join|subscribe|follow|click here|link in bio)\b',
        r'\b(اشترك|تابع|رابط|انضم|سجل)\b',
        #r'@\w+',             # Mentions
        r'#\w+',             # Hashtags
        #r'\d{10,}',          # Long numbers
        r'[\u2700-\u27BF]',  # Dingbats
        r'[\uE000-\uF8FF]',  # Private use area
    ]
    matches = sum(bool(re.search(pattern, text_without_emojis, re.IGNORECASE)) for pattern in spam_patterns)
    return matches >= 1  # At least 2 spam indicators  we can change it if we want to be more or less strict


def remove_mentions(text):

    if not isinstance(text, str):
        return text
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove Facebook-style tagged names (names that appear in comments as plain text)
    text = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', '', text)
    
    # Remove any remaining names in ALL CAPS (common in Arabic comments)
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

# Example usage:
# df['comment_message'] = df['comment_message'].apply(remove_mentions)

'''def analyze_sentiment(text, lang, sentiment_models):
    """Enhanced sentiment analysis with proper numerical stability"""
    if pd.isna(text) or text == '[null]' or not isinstance(text, str) or not text.strip():
        return {'label': 'neutral', 'score': 0.5}
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {'label': 'neutral', 'score': 0.5}

    # Load sentiment maps if not already loaded
    if not hasattr(analyze_sentiment, 'prayer_terms_map'):
        analyze_sentiment.prayer_terms_map, analyze_sentiment.emoji_sentiment_map = load_sentiment_maps()
    
    # Determine which model to use
    model_key = lang
    if lang not in sentiment_models or lang == 'unknown':
        model_key = 'ar' if re.search(r'[\u0600-\u06FF]', cleaned_text) else 'en'
    
    # Load or retrieve model from cache
    if model_key not in model_cache:
        model_name = sentiment_models[model_key]
        tokenizer, model = load_model(model_name)
        if tokenizer is None or model is None:
            return {'label': 'neutral', 'score': 0.5}
        model_cache[model_key] = (tokenizer, model)
    else:
        tokenizer, model = model_cache[model_key]
    
    try:
        # Tokenize input with proper truncation
        max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        
        # Get model outputs - ensure we're using PyTorch tensors
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert logits to numpy with proper numerical stability
        logits = outputs.logits
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy().astype(np.float64)
        else:
            # Handle TF tensors if needed
            import tensorflow as tf
            if tf.is_tensor(logits):
                logits_np = logits.numpy().astype(np.float64)
            else:
                logits_np = np.array(logits, dtype=np.float64)
        
        # Stable softmax calculation
        logits_np = logits_np - np.max(logits_np, axis=1, keepdims=True) # Subtract max for numerical stability
        exp_logits = np.exp(logits_np)
        scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        scores = scores[0]  # Get first batch
        predicted_label = int(np.argmax(scores))
        confidence_score = float(np.max(scores))

        scores = np.array(scores, dtype=np.float64).flatten()  # Ensure 1D float64 array
        if not isinstance(scores, (np.ndarray, list)) or len(scores) == 0:
            raise ValueError("Empty or invalid scores received for sentiment classification.")

        if len(scores) == 2:  # Binary classification
            sentiment = 'positive' if scores[1] > scores[0] else 'negative'
            sentiment_score = float(scores[1] if scores[1] > scores[0] else 1 - scores[1])

        elif len(scores) == 3:  # 3-class
            idx = int(np.argmax(scores))
            sentiment_labels = ['negative', 'neutral', 'positive']
            sentiment = sentiment_labels[idx]
            sentiment_score = float(scores[idx])

        elif len(scores) == 5:  # 5-class
            idx = int(np.argmax(scores))
            sentiment_labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
            if idx < 2:
                sentiment = 'negative'
                sentiment_score = 0.25 * (2 - idx)
            elif idx > 2:
                sentiment = 'positive'
                sentiment_score = 0.5 + 0.25 * (idx - 2)
            else:
                sentiment = 'neutral'
                sentiment_score = 0.5

        else:  # Fallback
            sentiment = 'neutral'
            sentiment_score = 0.5

        # Prayer-based adjustment
        prayer_score_adjustment = 0.0
        for term, term_score in analyze_sentiment.prayer_terms_map.items():
            if term.lower() in cleaned_text.lower():
                try:
                    weight = min(1.0, 10.0 / len(cleaned_text.split()))
                    prayer_score_adjustment += float(term_score) * weight
                except Exception as e:
                    logging.warning(f"Prayer term score error: {e}")

        # Emoji-based adjustment
        emoji_score_adjustment = 0.0
        for emoji_char in [c for c in cleaned_text if is_emoji(c)]:
            if emoji_char in analyze_sentiment.emoji_sentiment_map:
                try:
                    emoji_score_adjustment += float(analyze_sentiment.emoji_sentiment_map[emoji_char]) * 0.15
                except Exception as e:
                    logging.warning(f"Emoji score error: {e}")

        # Final sentiment score with bounds
        adjusted_score = float(sentiment_score) + prayer_score_adjustment + emoji_score_adjustment
        final_score = max(0.0, min(1.0, adjusted_score))

        # Final label
        if final_score < 0.4:
            final_sentiment = 'negative'
        elif final_score > 0.6:
            final_sentiment = 'positive'
        else:
            final_sentiment = 'neutral'

        return {'label': final_sentiment, 'score': final_score}

    except Exception as e:
        logging.error(f"[Sentiment] Error for lang={lang}, model={model_key}: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Text sample causing error: {cleaned_text[:50]}...")
        print(f"Error: {e}")
        return {'label': 'neutral', 'score': 0.5}
'''
QUESTION_PATTERNS = [
    r'\?',                    # Question mark
    r'^[آأ]',                 # Starts with آ or أ (Arabic)
    r'\b(do|does|did|is|are)\b',  # English inversion
    r'\b(est-ce que|avez-vous)\b', # French patterns
    r'\b(هل|أليس|ألا)\b',     # Arabic patterns
    r'\b(واش|هلا|اش)\b'       # Darija patterns
]
GLOBAL_QUESTION_WORDS = {
        'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whom', 'whose', 'can', 'could', 'may'
        'quoi', 'pourquoi', 'comment', 'quand', 'où', 'qui', 'quel', 'peut', 'pourrait'
        'ما', 'ماذا', 'لماذا', 'كيف', 'متى', 'أين', 'من', 'هل'
        'عندكم ','كم', 'كاين','علاش','علاه','منين','قداش','واش','وقتاش','اش', 'علاش', 'كيفاش', 'وقتاش', 'وين', 'شكون'
    }

# Global initialization
emoji_sentiment_map = load_emoji_terms_map()
prayer_terms_map = load_prayer_terms_map()

def detect_questions(text: str) -> Tuple[bool, Dict[str, bool]]:

    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return False, {}
    
    cleaned_text = clean_text(text)
    text_lower = cleaned_text.lower()
    indicators = {
        'question_mark': False,
        'question_word': False,
        'question_prefix': False,
        'interrogative_phrase': False
    }
    
    indicators['question_mark'] = any(
        mark in cleaned_text for mark in ['?', '؟', '⸮']  # English, Arabic question marks
    )
    GLOBAL_QUESTION_WORDS = {
        'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whom', 'whose', 'can', 'could', 'may'
        'quoi', 'pourquoi', 'comment', 'quand', 'où', 'qui est', 'quel', 'peut', 'pourrait'
        'متى','ماذا', 'لماذا', 'كيف', 'متى', 'أين','هل'
        'من فضلك','علاش','علاه','منين','قداش','واش','وقتاش','اش', 'علاش', 'كيفاش', 'وقتاش', 'وينتا', 'وين', 'شكون'
    }
    words_in_text = set(text_lower.split())
    indicators['has_question_word'] = not words_in_text.isdisjoint(GLOBAL_QUESTION_WORDS)
    # Check for language patterns
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            indicators['matches_pattern'] = True
            break
    # Check for sentence inversion (English/French)
    first_word = text_lower.split()[0] if text_lower.split() else ''
    indicators['is_inverted'] = first_word in {
        'do', 'does', 'did', 'is', 'are', 'can', 'could', 'would',
        'est-ce', 'avez', 'as', 'es', 'suis'
    }
    '''for lang, words in GLOBAL_QUESTION_WORDS.items():
        if any(word in text_lower for word in words):
            indicators['question_word'] = True
            break'''
    
    # Check for inversion/question prefixes (French/English)
    if cleaned_text.startswith(('est-ce que', 'is it', 'do you', 'did you', 'are you')):
        indicators['question_prefix'] = True
    
    # Arabic question patterns
    if re.search(r'(^|\s)(هل|أ|آ)\s', cleaned_text):
        indicators['interrogative_phrase'] = True
    
    is_question = any(indicators.values())
    
    return is_question, indicators
def analyze_questions(df):
    questions_df = df[df['is_question']]
    
    if questions_df.empty:
        print("No questions found in this batch")
        return None
    
    print(f"\n=== GLOBAL QUESTION ANALYSIS ===")
    print(f"Found {len(questions_df)} questions ({len(questions_df)/len(df)*100:.1f}% of total)")
    
    # Analyze detection patterns
    print("\nDetection Methods:")
    indicators = pd.json_normalize(questions_df['question_indicators'])
    #indicators = pd.json_normalize(questions_df['question_indicators'].apply(json.loads))
    for col in indicators.columns:
        print(f"  {col}: {indicators[col].sum()} ({indicators[col].mean()*100:.1f}%)")
    
    # Most common question words
    all_questions = ' '.join(questions_df['comment_message'].astype(str)).lower()
    question_word_counts = {
        word: all_questions.count(word)
        for word in GLOBAL_QUESTION_WORDS
        if word in all_questions
    }
    '''
    print("\nTop Question Words:")
    #for word, count in sorted(question_word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    for word, count in sorted(question_word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word}: {count}")
    '''
    
    # Sentiment comparison
    print("\nSentiment Comparison:")
    print("  Questions:")
    print(questions_df['sentiment_label'].value_counts(normalize=True).to_dict())
    print("\n  Non-questions:")
    print(df[~df['is_question']]['sentiment_label'].value_counts(normalize=True).to_dict())
    
    return {
        'total_questions': len(questions_df),
        'question_rate': len(questions_df)/len(df),
        'detection_methods': indicators.mean().to_dict(),
        'top_question_words': dict(sorted(question_word_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        'sentiment_comparison': {
            'questions': questions_df['sentiment_label'].value_counts(normalize=True).to_dict(),
            'non_questions': df[~df['is_question']]['sentiment_label'].value_counts(normalize=True).to_dict()
        }
    }

emoji_sentiment_map = load_emoji_terms_map()
def analyze_emoji_sentiment(text, emoji_sentiment_map):

    if emoji_sentiment_map is None:
        if hasattr(analyze_sentiment, 'emoji_sentiment_map'):
            emoji_sentiment_map = analyze_sentiment.emoji_sentiment_map
        else:
            return {'label': 'neutral', 'score': 0.5, 'spam': False}

    cleaned_text = text.strip()

    emoji_scores = []
    for char in cleaned_text:
        if is_emoji(char) and char in emoji_sentiment_map:
            try:
                score = float(emoji_sentiment_map[char])
                emoji_scores.append(score)
            except Exception as e:
                logging.warning(f"Emoji score error: {e}")

    if emoji_scores:
        avg_score = sum(emoji_scores) / len(emoji_scores)
        if avg_score < 0.4:
            sentiment = 'negative'
        elif avg_score > 0.6:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        return {'label': sentiment, 'score': avg_score, 'spam': False}
    else:
        return {'label': 'neutrall', 'score': 0.05, 'spam': False}


def analyze_sentiment(text, lang, sentiment_models, emoji_sentiment_map, prayer_terms_map):

    if not hasattr(analyze_sentiment, 'prayer_terms_map'):
        analyze_sentiment.prayer_terms_map = load_prayer_terms_map()

    if not hasattr(analyze_sentiment, 'emoji_sentiment_map'):
        analyze_sentiment.emoji_sentiment_map = load_emoji_terms_map()

    if is_emoji_only(text):
       return analyze_emoji_sentiment(text, emoji_sentiment_map)
      
    
    prayer_score_adjustment = 0.0
    emoji_score_adjustment = 0.0

     #this could neutralise emoji only comments !!!!!
    '''if pd.isna(text) or text == '[null]' or not isinstance(text, str) or not text.strip():
        return {'label': 'neutral', 'score': 0.5}'''
    # First check for spam
    if is_spam_comment(text):
        return {'label': None, 'score': None, 'spam': True}

    # Handle empty/invalid text
    if pd.isna(text) or text == '[null]' or not isinstance(text, str) or not text.strip():
        return {'label': 'neutral', 'score': 0.5}
    
    # Check for emoji-only comments
    cleaned_text = text.strip()
    is_emoji_only_flag = all(is_emoji(c) for c in cleaned_text if c.strip())
    
    # Emoji-only sentiment analysis
    if is_emoji_only_flag:
        emoji_scores = []
        for char in cleaned_text:
            if is_emoji(char) and char in analyze_sentiment.emoji_sentiment_map:
                try:
                    score = float(analyze_sentiment.emoji_sentiment_map[char])
                    emoji_scores.append(score)
                except Exception as e:
                    logging.warning(f"Emoji score error: {e}")
            
        if emoji_scores:
            avg_score = sum(emoji_scores) / len(emoji_scores)
            if avg_score < 0.4:
                sentiment = 'negative'
            elif avg_score > 0.6:
                sentiment = 'positive'
            else:
                sentiment = 'neutral'
            return {'label': sentiment, 'score': avg_score, 'spam': False}
        else:
            return {'label': 'neutral', 'score': 0.5, 'spam': False}
        
    '''
    # Si le texte contient uniquement des emojis connus
    if all(is_emoji(c) for c in text if c.strip()):
        emoji_score = 0.0
        count = 0
        for emoji_char in text:
            if emoji_char in analyze_sentiment.emoji_sentiment_map:
                try:
                    emoji_score += float(analyze_sentiment.emoji_sentiment_map[emoji_char])
                    count += 1
                except Exception as e:
                    logging.warning(f"Emoji score error: {e}")
        if count > 0:
            avg_score = emoji_score / count
            final_score = max(0.0, min(1.0, avg_score))
            if final_score < 0.4:
                sentiment = 'negative'
            elif final_score > 0.6:
                sentiment = 'positive'
            else:
                sentiment = 'neutral'
            return {'label': sentiment, 'score': final_score}
        
    '''
    cleaned_text = clean_text(text)

    cleaned_text = remove_mentions(cleaned_text)
    if not cleaned_text:
        return {'label': 'neutral', 'score': 0.5}
    
    # Handle Arabic text encoding issues
    if lang in ['ar', 'darija']:
        cleaned_text = cleaned_text.encode('utf-8').decode('utf-8', 'ignore')
     # Determine which model to use
    model_key = lang
    if lang not in sentiment_models or lang == 'unknown':
        model_key = 'ar' if re.search(r'[\u0600-\u06FF]', cleaned_text) else 'darija'
   
    if lang in ['ar', 'darija']:
        #sentiment_pipeline = pipeline(task="text-classification",model="Abdo36/Arabert-Sentiment-Analysis-ArSAS")
        sentiment_pipeline =pipeline(
                    task="text-classification",
                    model="/home/benslimane_m/scrapper/scrapper_2/Arabert-Sentiment-Offline/",
                    tokenizer="/home/benslimane_m/scrapper/scrapper_2/Arabert-Sentiment-Offline/",
                    truncation=True,         )
        result = sentiment_pipeline(cleaned_text)[0]
        return {'label': result['label'], 'score': result['score']}
    
    # Load or retrieve model from cache
    if model_key not in model_cache:
        model_name = sentiment_models[model_key]
        tokenizer, model = load_model(model_name)
        if tokenizer is None or model is None:
            return {'label': 'neutral', 'score': 0.5}
        model_cache[model_key] = (tokenizer, model)
    else:
        tokenizer, model = model_cache[model_key]
    
    try:
        # Tokenize input with proper truncation
        max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            return_token_type_ids=False 
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy().astype(np.float64)
        else:
            logits_np = np.array(logits, dtype=np.float64)
        
        # Stable softmax calculation
        logits_np = logits_np - np.max(logits_np, axis=1, keepdims=True) # Subtract max for numerical stability
        exp_logits = np.exp(logits_np)
        scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        scores = scores[0]  # Get first batch
        predicted_label = int(np.argmax(scores))
        confidence_score = float(np.max(scores))

        scores = np.array(scores, dtype=np.float64).flatten()  # Ensure 1D float64 array
        if not isinstance(scores, (np.ndarray, list)) or len(scores) == 0:
            raise ValueError("Empty or invalid scores received for sentiment classification.")

        if len(scores) == 2:  # Binary classification
            sentiment = 'positive' if scores[1] > scores[0] else 'negative'
            sentiment_score = float(scores[1] if scores[1] > scores[0] else 1 - scores[1])

        elif len(scores) == 3:  # 3-class
            idx = int(np.argmax(scores))
            sentiment_labels = ['negative', 'neutral', 'positive']
            sentiment = sentiment_labels[idx]
            sentiment_score = float(scores[idx])

        elif len(scores) == 5:  # 5-class
            idx = int(np.argmax(scores))
            sentiment_labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
            if idx < 2:
                sentiment = 'negative'
                sentiment_score = 0.25 * (2 - idx)
            elif idx > 2:
                sentiment = 'positive'
                sentiment_score = 0.5 + 0.25 * (idx - 2)
            else:
                sentiment = 'neutral'
                sentiment_score = 0.5

        else:  # Fallback
            sentiment = 'neutral'
            sentiment_score = 0.5
        
        # Prayer-based adjustment
        
        for term, score in analyze_sentiment.prayer_terms_map.items():
            if term.lower() in cleaned_text.lower():
                try:
                    weight = min(1.0, 10.0 / len(cleaned_text.split()))
                    prayer_score_adjustment += float(score) * weight
                except Exception as e:
                    logging.warning(f"Prayer term score error: {e}") 
        # Emoji-based adjustment
        for emoji_char in [c for c in cleaned_text if is_emoji(c)]:
            if emoji_char in analyze_sentiment.emoji_sentiment_map:
                try:
                   #emoji_score_adjustment += float(analyze_sentiment.emoji_sentiment_map[emoji_char]) * 0.15
                    sentiment_label = analyze_sentiment.emoji_sentiment_map[emoji_char]
                    if sentiment_label == 'positive':
                        emoji_score_adjustment += 0.15
                    elif sentiment_label == 'negative':
                        emoji_score_adjustment -= 0.15
                    elif sentiment_label == 'neutral':
                        emoji_score_adjustment += 0.0
            # Final sentiment score with bounds
                except Exception as e:
                    logging.warning(f"Emoji score error: {e}")
        # Final sentiment score with bounds
        adjusted_score = float(sentiment_score) + prayer_score_adjustment + emoji_score_adjustment
        final_score = max(0.0, min(1.0, adjusted_score))

        if final_score < 0.4:
            final_sentiment = 'negative'
        elif final_score > 0.6:
            final_sentiment = 'positive'
        else:
            final_sentiment = 'neutral'
        return {'label': final_sentiment, 'score': final_score, 'spam': False  # Explicitly mark as not spam
        }
        
    except Exception as e:
        logging.error(f"[Sentiment] Error for lang={lang}, model={model_key}: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Text sample causing error: {cleaned_text[:50]}...")
        print(f"Error: {e}")
        return {'label': 'neutral', 'score': 0.5,
            'spam': False }
     
def get_stopwords(lang):
    try:
        if lang == 'en':
            return list(stopwords.words('english'))
        elif lang == 'fr':
            return list(stopwords.words('french'))
        elif lang == 'ar':
            arabic_stops = set(stopwords.words('arabic'))
            arabic_stop = [
                'في', 'من', 'إلى', 'على', 'أن', 'ما', 'هذه', 'يكون',
                'في', 'من', 'الى', 'عن', 'على', 'ان', 'لا', 'ما', 'هذا', 'هذه', 
                'كان', 'هو', 'هي', 'هم', 'إذ', 'إذا', 'ذلك',
                'التي', 'الذي', 'الذين',
            ]
            combined_arabic_stops = list(arabic_stops.union(arabic_stop))
            vectorizer = CountVectorizer(stop_words=combined_arabic_stops)     
            return vectorizer
        elif lang == 'darija':
            arabic_stops = list(stopwords.words('arabic'))
            darija_stops = ['wesh', 'kifesh', 'kayen', 'ana', 'nta', 'hna', 'had', 'dyal', 'dyali', 'dyalna']
            return arabic_stops + darija_stops
        else:
            return []
    
    except Exception as e:
        print(f"Error getting stopwords for language {lang}: {e}")
        return []
def extract_topics(df, lang):
    lang_df = df[df['detected_language'] == lang]
    
    if lang_df.empty or len(lang_df) < 5:
        return None
    
    # Get language-specific stopwords
    stop_words = get_stopwords(lang)
    
    # Additional preprocessing for Arabic
    if lang in ['ar', 'darija']:
        def preprocess_arabic(text):
            text = re.sub(r'[إأٱآا]', 'ا', text)  # alef
            text = re.sub(r'[ى]', 'ي', text)      # yea
            text = re.sub(r'[ؤئ]', 'ء', text)     # hamza
            return text
        
        texts = lang_df['comment_message'].fillna('').apply(clean_text).apply(preprocess_arabic).tolist()
    else:
        texts = lang_df['comment_message'].fillna('').apply(clean_text).tolist()
    
    try:
        # Create vectorizer with consistent token pattern
        vectorizer = CountVectorizer(
            max_df=0.9,
            min_df=3,
            max_features=1000,
            preprocessor=clean_text,
            stop_words=stop_words if stop_words else None,
            token_pattern=r'\b[^\d\W]+\b' 
        )
        
        dtm = vectorizer.fit_transform(texts)
        n_topics = min(5, max(2, len(texts) // 20))
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            random_state=42,
            max_iter=25
        )
        lda.fit(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_concentration = sum(topic[i] for i in top_indices) / sum(topic)
            
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'coherence': round(float(topic_concentration), 3)
            })
        
        return topics
    except Exception as e:
        print(f"Topic extraction error for language {lang}: {e}")
        return None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_analysis_results(results):

    detailed_file = 'sentiment_analysis_results.csv'
    summary_file = 'sentiment_analysis_summary.json'
    
    detailed_df = pd.DataFrame(results['detailed'])
    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8')
    print(f"Saved detailed results to {detailed_file}")
    
    summary_json = {
        'total_comments': int(results['summary']['total_comments']),
        'valid_comments': int(results['summary']['valid_comments']),
        'language_distribution': {
            k: int(v) for k, v in results['summary']['language_distribution'].items()
        },
        'sentiment_distribution': {
            k: int(v) for k, v in results['summary']['sentiment_distribution'].items()
        },
        'avg_sentiment_score_by_language': {
            k: float(v) for k, v in results['summary']['avg_sentiment_score_by_language'].items()
        },
    }
    
    summary_json['topics_by_language'] = {}
    for lang, topics in results['summary']['topics_by_language'].items():
        if topics is not None:
            summary_json['topics_by_language'][lang] = [
                {
                    'topic_id': int(topic['topic_id']),
                    'top_words': list(topic['top_words']),
                    'coherence': float(topic['coherence'])
                }
                for topic in topics
            ]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_file}")
    '''
def analyze_facebook_comments():

    all_results = {
        'detailed': [],
        'summary': {
            'total_comments': 0,
            'valid_comments': 0,
            'language_distribution': {},
            'sentiment_distribution': {},
            'avg_sentiment_score_by_language': {},
            'topics_by_language': {},
            'question_analysis': {}
        }
    }
    batch_count = 0
    print("Starting Facebook comment analysis...")
   
    while True:
        batch_count += 1
        print(f"\nProcessing batch {batch_count}...")
        df = extract_comments()
        if df is None or len(df) == 0:
            print("No more comments to process")
            break 
        print(f"Extracted {len(df)} comments from database")
        
        df['comment_message'] = df['comment_message'].replace('[null]', np.nan)
        # hna we can add more cleaning if needed
        df['clean_message'] = df['comment_message'].apply(clean_text).str.lower()
        df = df[df['clean_message'].str.strip().str.len() > 0]
        valid_comments = df['clean_message'].apply(lambda x: len(str(x).strip()) > 0).sum()
        print(f"Valid comments in batch: {len(df)}")
        print(f"Valid comments with content: {valid_comments}")
        
        print("Detecting languages...")
        df['detected_language'] = df['comment_message'].apply(detect_language)
        
        lang_counts = df['detected_language'].value_counts()
        print("Language distribution:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count} comments ({count/len(df)*100:.1f}%)")
        
        print("Loading sentiment models...")
        sentiment_models = load_sentiment_models()
        prayer_terms_map, emoji_sentiment_map = load_sentiment_maps()
        analyze_sentiment.prayer_terms_map = prayer_terms_map
        analyze_sentiment.emoji_sentiment_map = emoji_sentiment_map
        
        print("Analyzing sentiment...")
        df['sentiment'] = df.apply(
            lambda row: analyze_sentiment(row['comment_message'], row['detected_language'], sentiment_models), 
            axis=1
        )
        
        df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'] if isinstance(x, dict) else 'neutral')
        df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'] if isinstance(x, dict) else 0.5)
        df['sentiment_label'] = df['sentiment_label'].str.lower()

        sentiment_counts = df['sentiment_label'].value_counts()
        print("Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} comments ({count/len(df)*100:.1f}%)")
        
        update_database(df)
        print("Extracting topics by language...")
        languages = df['detected_language'].unique()
        topics_by_language = {}
        for lang in languages:
            if lang in ['en', 'fr', 'ar', 'darija']:
                topics = extract_topics(df, lang)
                if topics:
                    topics_by_language[lang] = topics
                    print(f"Extracted {len(topics)} topics for language {lang}")
                else:
                    print(f"No topics extracted for language {lang}")
            else:
                print(f"Skipping unsupported language: {lang}")
        
        try:
            with open('topics_by_language.json', 'w', encoding='utf-8') as f:
                json.dump(topics_by_language, f, ensure_ascii=False, cls=NumpyEncoder)
            print("Topics saved to topics_by_language.json")
        except Exception as e:
            print(f"Error saving topics to JSON: {e}")
        
        results = {
            'detailed': df.to_dict(orient='records'),
            'summary': {
                'total_comments': len(df),
                'valid_comments': valid_comments,
                'language_distribution': df['detected_language'].value_counts().to_dict(),
                'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
                'avg_sentiment_score_by_language': df.groupby('detected_language')['sentiment_score'].mean().to_dict(),
                'topics_by_language': {lang: extract_topics(df, lang) for lang in df['detected_language'].unique()}
            }
        }

        results = analyze_facebook_comments()
        question_stats = analyze_questions(results['detailed'])
        results['summary']['question_analysis'] = question_stats
        save_analysis_results(results)

        question_stats = analyze_questions(df)
        if question_stats:
            results['summary']['question_analysis'] = question_stats
        generate_analysis_report(results)
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total comments processed: {len(df)}")
        print(f"Languages detected: {', '.join(df['detected_language'].unique())}")
        print(f"Sentiment distribution: {dict(df['sentiment_label'].value_counts())}")
        print(f"Results saved to: sentiment_analysis_results.csv and sentiment_analysis_summary.json")
       
        print("Analysis completed successfully!")
        return "Analysis completed successfully!"
'''

def analyze_facebook_comments():
    empty_batch_count = 0
    max_empty_batches = 5
    print("Starting Facebook comment analysis...")
    
    all_results = {
        'detailed': [],
        'summary': {
            'total_comments': 0,
            'valid_comments': 0,
            'language_distribution': {},
            'sentiment_distribution': {},
            'avg_sentiment_score_by_language': {},
            'topics_by_language': {},
            'question_analysis': {}
        }
    }

    print("Loading sentiment models...")
    sentiment_models = load_sentiment_models()
    #prayer_terms_map, emoji_sentiment_map = load_sentiment_maps()
    analyze_sentiment.prayer_terms_map =  load_prayer_terms_map()
    analyze_sentiment.emoji_sentiment_map = load_emoji_terms_map()

    batch_count = 0
    while True:
        batch_count += 1
        print(f"\nProcessing batch {batch_count}...")
        df = extract_comments()
        
        if df is None or len(df) == 0:
            empty_batch_count += 1
            print(f"No comments extracted (null or empty). Empty batch count: {empty_batch_count}/{max_empty_batches}")
            if empty_batch_count >= max_empty_batches:
                print("No more comments to process")
                break
            continue 

        # Data cleaning
        df['is_emoji_only'] = df['comment_message'].apply(is_emoji_only)
        df['comment_message'] = df['comment_message'].replace('[null]', np.nan)
        # hna i can add more cleaning if needed
        df['is_spam'] = df['comment_message'].apply(is_spam_comment)
        df['cleaned_message'] = df['comment_message'].apply(clean_text).str.lower()
        #df = df[df['comment_message'].str.strip().str.len() > 0]
        #df = df[df['comment_message'].apply(lambda x: isinstance(x, str) and (x.strip() != '' or any(is_emoji(c) for c in x)))]
        df['is_valid'] = ~df['is_spam'] & df['comment_message'].apply(is_valid_comment)
        df['is_valid'] = ~df['is_spam'] & df['cleaned_message'].apply(
            lambda x: isinstance(x, str) and (x.strip() != '' or any(is_emoji(c) for c in x))
        )

        #valid_comments = len(df)
        valid_df = df[df['is_valid']].copy()
        #print(f"[Batch {batch_count}] Extracted: {len(df)} | Valid after cleaning: {valid_comments}")
        print(f"Extracted {len(df)} comments from database")

        '''if len(df[df['comment_message'].str.strip().astype(bool)]) == 0:
            empty_batch_count += 1
            if empty_batch_count >= max_empty_batches:
                print("Stopping - too many consecutive batches with no valid comments")
                break
            continue
        else:
            empty_batch_count = 0
        '''
        '''
        if valid_comments == 0:
            empty_batch_count += 1
            print(f"No valid comments found in batch {batch_count} ({empty_batch_count}/{max_empty_batches})")
            if empty_batch_count >= max_empty_batches:
                print("Stopping - too many consecutive batches with no valid comments")
                break
            continue
        else:
            empty_batch_count = 0

        #valid_comments = df['comment_message'].apply(lambda x: len(str(x).strip()) > 0).sum()
        print(f"Valid comments in batch: {len(df)}")
        print(f"Valid comments with content: {valid_comments}")
        '''
        if len(valid_df) == 0:
            empty_batch_count += 1
            print(f"No valid comments in batch {batch_count} ({empty_batch_count}/{max_empty_batches})")
            if empty_batch_count >= max_empty_batches:
                print("Stopping - too many consecutive batches with no valid comments")
                break
            continue
        else:
            empty_batch_count = 0

        print(f"Processing {len(valid_df)} valid comments (of {len(df)} total)")

        # Language detection
        print("Detecting languages...")
        #valid_df['detected_language'] = valid_df['cleaned_message'].apply(detect_language)
        valid_df['detected_language'] = valid_df.apply(
            lambda row: 'emoji' if row['is_emoji_only'] else detect_language(row['cleaned_message']),
            axis=1
        )

        # Question detection
        print("Detecting questions...")
        #question_results = df['comment_message'].apply(detect_questions)
        question_results = valid_df['cleaned_message'].apply(detect_questions)
        valid_df['is_question'] = [result[0] for result in question_results]
        valid_df['question_indicators'] = [result[1] for result in question_results]
        
        # Sentiment analysis
       
        print("Analyzing sentiment...")
        '''
        valid_df['sentiment'] = valid_df.apply(
            lambda row: analyze_sentiment(row['cleaned_message'], row['detected_language'], sentiment_models, emoji_sentiment_map, prayer_terms_map),
            axis=1
        )
        '''
        valid_df['sentiment'] = valid_df.apply(
            lambda row: analyze_emoji_sentiment(row['comment_message'], emoji_sentiment_map)
            if row['is_emoji_only']
            else analyze_sentiment(row['cleaned_message'], row['detected_language'], sentiment_models, emoji_sentiment_map, prayer_terms_map),
            axis=1
        )
        '''
        df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'].lower() if isinstance(x, dict) else 'neutral')
        df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'] if isinstance(x, dict) else 0.5)
        df['is_spam'] = df['sentiment'].apply(lambda x: x.get('spam', False) if isinstance(x, dict) else False)
        df.loc[df['is_spam'], 'sentiment_label'] = None
        df.loc[df['is_spam'], 'sentiment_score'] = None 
    #   df_filtered = df[~df['is_spam']]
        '''
        # Extract sentiment results
        valid_df['sentiment_label'] = valid_df['sentiment'].apply(
            lambda x: x['label'].lower() if isinstance(x, dict) else 'neutral'
        )
        valid_df['sentiment_score'] = valid_df['sentiment'].apply(
            lambda x: x['score'] if isinstance(x, dict) else 0.5
        )
        valid_df['is_spam'] = valid_df['sentiment'].apply(lambda x: x.get('spam', False) if isinstance(x, dict) else False)
        valid_df.loc[valid_df['is_spam'], 'sentiment_label'] = None
        valid_df.loc[valid_df['is_spam'], 'sentiment_score'] = None 
        spam_df = df[df['is_spam']].copy()
        update_database(pd.concat([valid_df, spam_df], ignore_index=True))

    #   df_filtered = df[~df['is_spam']]
        print("DEBUG: Columns in final valid_DF:", valid_df.columns.tolist())
        print("DEBUG: Sample row:", valid_df.iloc[0].to_dict())

        # Merge results back to original dataframe
        df.update(valid_df)
        
        # Ensure spam comments are properly marked
        df.loc[df['is_spam'], 'sentiment_label'] = None
        df.loc[df['is_spam'], 'sentiment_score'] = None

        #update_database(valid_df)
        print("DEBUG: Columns in final DF:", df.columns.tolist())
        print("DEBUG: Sample row:", df.iloc[0].to_dict())

        # Topic extraction
        print("Extracting topics by language...")
        topics_by_language = {}
        for lang in ['en', 'fr', 'ar', 'darija']:
            if lang in valid_df['detected_language'].unique():
                    lang_comments = valid_df[valid_df['detected_language'] == lang]
                    topics = extract_topics(lang_comments, lang)
                    if topics:
                        topics_by_language[lang] = topics
                        print(f"  Extracted {len(topics)} topics for {lang}")   

        #update_database(df)
        
        # Aggregate results
        batch_results = {
            'detailed': valid_df.to_dict(orient='records'),
            'summary': {
                'total_comments': len(df),
                'valid_comments': len(valid_df),
                'language_distribution': valid_df['detected_language'].value_counts().to_dict(),
                'sentiment_distribution': valid_df['sentiment_label'].value_counts().to_dict(),
                'avg_sentiment_score_by_language': valid_df.groupby('detected_language')['sentiment_score'].mean().to_dict(),
                'topics_by_language': topics_by_language,
                'question_analysis': analyze_questions(valid_df) if len(valid_df) > 0 else None
            }
        }       
        # Merge batch results with overall results
        all_results['detailed'].extend(batch_results['detailed'])
        for key in ['total_comments', 'valid_comments']:
            all_results['summary'][key] += batch_results['summary'][key]
        
        for key in ['language_distribution', 'sentiment_distribution']:
            for k, v in batch_results['summary'][key].items():
                all_results['summary'][key][k] = all_results['summary'][key].get(k, 0) + v
        
        for lang, score in batch_results['summary']['avg_sentiment_score_by_language'].items():
            if lang not in all_results['summary']['avg_sentiment_score_by_language']:
                all_results['summary']['avg_sentiment_score_by_language'][lang] = score
            else:
                # Weighted average for scores
                total = all_results['summary']['language_distribution'][lang] + batch_results['summary']['language_distribution'].get(lang, 0)
                if total > 0:
                    all_results['summary']['avg_sentiment_score_by_language'][lang] = (
                        (all_results['summary']['avg_sentiment_score_by_language'][lang] * all_results['summary']['language_distribution'][lang] +
                         score * batch_results['summary']['language_distribution'].get(lang, 0)) / total
                    )
        # Merge topics
        for lang, topics in batch_results['summary']['topics_by_language'].items():
            if lang not in all_results['summary']['topics_by_language']:
                all_results['summary']['topics_by_language'][lang] = topics
        
        # Merge question analysis
        if batch_results['summary']['question_analysis']:
            if not all_results['summary']['question_analysis']:
                all_results['summary']['question_analysis'] = batch_results['summary']['question_analysis']
            else:
                # Combine question stats
                qa = all_results['summary']['question_analysis']
                batch_qa = batch_results['summary']['question_analysis']
                
                qa['total_questions'] += batch_qa['total_questions']
                qa['question_rate'] = qa['total_questions'] / all_results['summary']['total_comments']
                
                for method, value in batch_qa['detection_methods'].items():
                    if method in qa['detection_methods']:
                        qa['detection_methods'][method] = (qa['detection_methods'][method] + value) / 2
                
                for word, count in batch_qa['top_question_words'].items():
                    qa['top_question_words'][word] = qa['top_question_words'].get(word, 0) + count

    print("\n=== FINAL ANALYSIS SUMMARY ===")
    print(f"Total comments processed: {all_results['summary']['total_comments']}")
    print(f"Languages detected: {', '.join(all_results['summary']['language_distribution'].keys())}")
    print("Sentiment distribution:")
    for sentiment, count in all_results['summary']['sentiment_distribution'].items():
        print(f"  {sentiment}: {count} comments ({count/all_results['summary']['total_comments']*100:.1f}%)")
    
    if all_results['summary']['question_analysis']:
        print("\nQuestion Analysis:")
        qa = all_results['summary']['question_analysis']
        print(f"  Total questions: {qa['total_questions']} ({qa['question_rate']*100:.1f}%)")
        print("  Detection methods:")
        for method, rate in qa['detection_methods'].items():
            print(f"    {method}: {rate*100:.1f}%")
        print("  Top question words:")
        for word, count in sorted(qa['top_question_words'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {word}: {count}")

    # Save final results
    save_analysis_results(all_results)
    generate_strategic_report_with_comments(df)
    
    print("\nAnalysis completed successfully!")
    return all_results

def generate_strategic_report_with_comments(df):
    
    print("\nGenerating strategic improvement report with raw comments...")

    # 1️⃣ Filtrage selon tes critères
    # Vérifie la présence des colonnes et complète si besoin
    required_columns = ['comment_message', 'post_id', 'detected_language', 'sentiment_label', 'is_question', 'commenter_name', 'is_spam']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # Ajoute la colonne manquante avec valeurs None

    # Filtrage selon tes critères (après ajout sécurisé des colonnes)
    filtered_df = df[
        (df['commenter_name'] != 'Hasnaoui Private Hospital') &
        (df['is_spam'] != True)
    ][['comment_message', 'post_id', 'detected_language', 'sentiment_label', 'is_question']].copy()

    comment_records = filtered_df.to_dict(orient='records')

    # EXTRACTION DES QUESTIONS
    questions_df = filtered_df[filtered_df['is_question'] == True].copy()
    questions_list = questions_df['comment_message'].tolist()

    # Conversion en liste de dictionnaires pour l'API
    comment_records = filtered_df.to_dict(orient='records')

    # 2️⃣ Conversion en liste de dictionnaires (plus propre pour LLM)
    comment_records = filtered_df.to_dict(orient='records')

    # 3️⃣ Construction du prompt
    prompt = f"""
You are a multilingual expert in marketing strategy, customer sentiment analysis, and customer experience optimization.

You will analyze real Facebook customer comments collected in multiple languages: French, English, Arabic and Algerian Arabic (Darija).

Here is a dataset of customer comments (message, language, sentiment label, question detection and post id context):

{json.dumps(comment_records, indent=2, ensure_ascii=False)}

Here is a list of questions frequently asked by customers:
{json.dumps(questions_list, indent=2, ensure_ascii=False)}

Based on this data, please generate:
1️⃣ Marketing strategy recommendations.
2️⃣ Customer engagement improvements.
3️⃣ Content and communication optimizations.
4️⃣ Language-specific or cultural insights.
5️⃣ Suggestions for potential answers or communication scripts to address the most frequently asked questions.
"""

    # 4️⃣ Construction de la requête API OpenRouter
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    api_key =  "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" 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        #"model": "mistralai/mistral-7b-instruct",
        "model": "opengvlab/internvl3-14b:free",
        "messages": [
            {"role": "system", "content": "You are a strategic marketing expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    # 5️⃣ Appel de l'API
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        generated_text = result['choices'][0]['message']['content']
        print("\n=== MARKETING RECOMMENDATIONS ===\n")
        print(generated_text)

        report_filename = f"marketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nReport saved to {report_filename}")

    except requests.exceptions.RequestException as e:
        print("API request failed:", e)
# Run the analysis
if __name__ == "__main__":
    analyze_facebook_comments()