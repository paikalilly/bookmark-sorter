# === mobileapp.py (Corrected for Model Amnesia Bug) ===

import streamlit as st
import pandas as pd
import joblib
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder # Import new encoders
from sklearn.preprocessing import MultiLabelBinarizer
import requests
import base64

def push_to_github(file_path, repo_file_path):
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets["GITHUB_BRANCH"]
    username = st.secrets["GITHUB_USERNAME"]

    with open(file_path, "rb") as f:
        content = f.read()
        encoded = base64.b64encode(content).decode("utf-8")

    # Check if file exists (to get SHA)
    get_url = f"https://api.github.com/repos/{repo}/contents/{repo_file_path}"
    headers = {"Authorization": f"token {token}"}
    r = requests.get(get_url, headers=headers)
    if r.status_code == 200:
        sha = r.json()["sha"]
    else:
        sha = None

    data = {
        "message": f"Update {repo_file_path} via Streamlit app",
        "content": encoded,
        "branch": branch
    }
    if sha:
        data["sha"] = sha

    res = requests.put(get_url, headers=headers, json=data)

    if res.status_code in [200, 201]:
        st.success(f"✅ Synced `{repo_file_path}` to GitHub.")
    else:
        st.error("❌ GitHub sync failed.")
        st.error(res.json())


# --- Page Config ---
st.set_page_config(layout="centered")

# --- Style tweaks for mobile ---
st.markdown("""
    <style>            
    html, body, [class*="css"]  {
        font-size: 16px !important;
    }
    h1 {
        font-size: 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }        
    input, textarea {
        font-size: 16px !important;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Override multiselect tag chip background */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #e0f7fa !important;
        color: #006064 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
      overflow: visible !important;
    }
    div[data-baseweb="popover"] {
      z-index: 9999 !important;
      transform-origin: bottom !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Config ---
MASTER_BASELINE_FILE = 'bookmarks_baseline.csv'
REMAINING_BOOKMARKS_FILE = 'bookmarks_remaining.csv'
NEWLY_SORTED_DATA_FILE = 'bookmarks_sorted.csv'
TAG_MODEL_FILE = 'tag_model.pkl'
TAG_BINARIZER_FILE = 'tag_mlb.pkl'
FOLDER_MODEL_FILE = 'folder_model.pkl'
FOLDER_ENCODER_FILE = 'folder_encoder.pkl'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- Caching Functions ---
@st.cache_resource
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def load_initial_data():
    try:
        return pd.read_csv(REMAINING_BOOKMARKS_FILE, sep=';').fillna('')
    except FileNotFoundError:
        st.error(f"ERROR: '{REMAINING_BOOKMARKS_FILE}' not found.")
        st.stop()

# --- Helper Functions ---
def get_embedding(text, model):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.reshape(1, -1)

def get_combined_data():
    """Loads and combines your master baseline data with new interactive data."""
    try:
        baseline_df = pd.read_csv(MASTER_BASELINE_FILE).fillna('')
    except FileNotFoundError:
        st.error(f"FATAL: Your master baseline file '{MASTER_BASELINE_FILE}' is missing.")
        return None
        
    if os.path.exists(NEWLY_SORTED_DATA_FILE):
        newly_sorted_df = pd.read_csv(NEWLY_SORTED_DATA_FILE).fillna('')
        return pd.concat([baseline_df, newly_sorted_df], ignore_index=True)
    return baseline_df

def full_retrain_all_models(embedding_model):
    """Performs the SLOW, full retraining of all models on ALL data."""
    st.info("⏳ Starting full model retraining... This may take a minute.")
    
    with st.spinner("Combining all known data..."):
        combined_df = get_combined_data()
        if combined_df is None: return

    with st.spinner("Retraining folder model..."):
        folder_encoder = LabelEncoder()
        folder_encoder.fit(combined_df['folder'])
        X_folder_embed = np.vstack([
            get_embedding(f"{r['note']} {r['tags']}", embedding_model)
            for _, r in combined_df.iterrows()
        ])
        y_folder_labels = folder_encoder.transform(combined_df['folder'])
        folder_model = SGDClassifier(loss='log_loss', random_state=42)
        folder_model.fit(X_folder_embed, y_folder_labels)
        st.session_state.folder_model, st.session_state.folder_encoder = folder_model, folder_encoder
        joblib.dump(folder_model, FOLDER_MODEL_FILE)
        joblib.dump(folder_encoder, FOLDER_ENCODER_FILE)

    with st.spinner("Retraining tag model..."):
        tag_mlb = MultiLabelBinarizer()
        all_tags_list = combined_df['tags'].apply(lambda x: [t.strip() for t in str(x).split(',') if t.strip()])
        tag_mlb.fit(all_tags_list)
        X_tags_embed = np.vstack([
            get_embedding(f"{r['note']} {r['folder']}", embedding_model)
            for _, r in combined_df.iterrows()
        ])
        y_tags = tag_mlb.transform(all_tags_list)
        tag_model = OneVsRestClassifier(SGDClassifier(loss='hinge', random_state=42))
        tag_model.fit(X_tags_embed, y_tags)
        st.session_state.tag_model, st.session_state.tag_mlb = tag_model, tag_mlb
        joblib.dump(tag_model, TAG_MODEL_FILE)
        joblib.dump(tag_mlb, TAG_BINARIZER_FILE)
        
    st.success("✅ All models fully retrained and updated!")

# --- Initialize Models and State ---
if 'initialized' not in st.session_state:
    st.session_state.embedding_model = load_embedding_model()
    try:
        st.session_state.tag_model = joblib.load(TAG_MODEL_FILE)
        st.session_state.tag_mlb = joblib.load(TAG_BINARIZER_FILE)
        st.session_state.folder_model = joblib.load(FOLDER_MODEL_FILE)
        st.session_state.folder_encoder = joblib.load(FOLDER_ENCODER_FILE)
    except FileNotFoundError:
        st.error("❌ Model files missing. Please run your training script first.")
        st.stop()
    
    st.session_state.new_items_staged_for_retrain = 0
    st.session_state.initialized = True


# --- Main App Logic ---
if 'df_full' in st.session_state:
    df_full = st.session_state['df_full']
else:
    df_full = load_initial_data()

# === 📂 FOLDER FILTER UI ===
# Get folders
available_folders = sorted(df_full['current_folder'].dropna().unique())
folder_options = ["— All —"] + available_folders

# --- Single selectbox ---
selected_folder = st.selectbox(
    "Select a folder to focus on:",
    options=folder_options,
    index=folder_options.index(
        st.session_state.get('selected_folder', "— All —")
    ) if st.session_state.get('selected_folder', "— All —") in folder_options else 0,
    key="folder_selector"
)

st.session_state['selected_folder'] = selected_folder

# --- Clear stale state if switching to All
if selected_folder == "— All —":
    st.session_state['last_selected_folder'] = "— All —"

# --- Reset states if folder changed ---
if 'last_selected_folder' not in st.session_state:
    st.session_state['last_selected_folder'] = selected_folder

if st.session_state['last_selected_folder'] != selected_folder:
    for key in ['note_input_area', 'folder_input', 'tags_input', 'new_folder_input', 'new_tag_input']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state['last_selected_folder'] = selected_folder

# --- Filter rows AFTER handling state ---
if selected_folder == "— All —":
    df = df_full.copy().reset_index(drop=True)
else:
    df = df_full[df_full['current_folder'] == selected_folder].reset_index(drop=True)

# Show info
st.info(f"📌 Working on **{len(df)}** bookmark(s) in: `{selected_folder}`")

# Handle empty
if df.empty:
    st.success(f"🎉 All bookmarks{' for folder ' + selected_folder if selected_folder != '— All —' else ''} are sorted!")
    st.balloons()
    st.stop()

row = df.iloc[0]
url = row.get('url', '')
current_folder = row.get('current_folder', '')
ai_suggested_folder = row.get('ai_suggested_folder', '')
current_tags = row.get('tags', '')
confidence = row.get('confidence', '')

# --- UI ---
st.title("AI Bookmark Sorter")
st.markdown(f"**URL:** [{url}]({url})")
st.markdown(f"**Confidence Score:** {confidence:.3f}")
st.markdown(f"Current Folder: {current_folder}")
st.markdown(f"Current Tags: {current_tags}")

# --- Note Input Area ---
note_key = "note_input_area"
clear_flag = "clear_note_now"

note_key = "note_input_area"
if note_key not in st.session_state:
    st.session_state[note_key] = row.get('note', '')

if st.session_state.get(clear_flag):
    st.session_state[note_key] = ""
    st.session_state[clear_flag] = False

if note_key not in st.session_state:
    st.session_state[note_key] = row.get('note', '')

note_col, clear_note_col = st.columns([5, 1])
with note_col:
    st.text_area(
        "Edit Note:",
        key=note_key,
        height=100
    )
with clear_note_col:
    if st.button("X", key="clear_note_button"):
        st.session_state[clear_flag] = True
        st.rerun()

current_note = st.session_state[note_key]

# --- Predict Folder (runs on every load) ---
text_for_folder = f"{current_note} {current_tags} {current_folder} {ai_suggested_folder}"
X_folder_vec = get_embedding(
    text_for_folder,
    st.session_state.embedding_model
)
predicted_folder_idx = st.session_state.folder_model.predict(X_folder_vec)
predicted_folder = st.session_state.folder_encoder.inverse_transform(predicted_folder_idx)[0]

folder_key = "folder_input"
new_folder_key = "new_folder_input"
folder_clear_flag = "clear_folder_now"

if st.session_state.get(folder_clear_flag):
    st.session_state[folder_key] = "—" # Default empty value for selectbox
    st.session_state[new_folder_key] = ""
    st.session_state[folder_clear_flag] = False # Unset the flag

if folder_key not in st.session_state:
    st.session_state[folder_key] = predicted_folder
if new_folder_key not in st.session_state:
    st.session_state[new_folder_key] = ""

folder_options = ["—"] + sorted(list(st.session_state.folder_encoder.classes_))

folder_input_col, clear_folder_col = st.columns([5, 1])
with folder_input_col:
    folder_input = st.selectbox(
        "Choose or confirm folder:",
        options=folder_options,
        index=folder_options.index(st.session_state[folder_key]) if st.session_state[folder_key] in folder_options else 0
    )
    new_folder_input = st.text_input(
        "Or type a new folder:",
        value=st.session_state.get(new_folder_key, "")
    )
with clear_folder_col:
    if st.button("X", key="clear_folder_button"):
        st.session_state[folder_clear_flag] = True
        st.rerun()

st.session_state[folder_key] = folder_input
st.session_state[new_folder_key] = new_folder_input

if new_folder_input.strip():
    final_folder = new_folder_input.strip()
elif folder_input != "—":
    final_folder = folder_input
else:
    final_folder = ""

# --- Predict Tags ONCE ---
if 'predicted_tags' not in st.session_state:
    text_for_tags = f"{current_note} {final_folder}"
    X_tags_vec = get_embedding(text_for_tags, st.session_state.embedding_model)
    predicted_tags_binary = st.session_state.tag_model.predict(X_tags_vec)
    predicted_tags = list(
        st.session_state.tag_mlb.inverse_transform(predicted_tags_binary)[0]
    )
    st.session_state['predicted_tags'] = predicted_tags
else:
    predicted_tags = st.session_state['predicted_tags']

# --- Tag Input State Keys ---
tags_key = "tags_input"
new_tag_key = "new_tag_input"
tags_clear_flag = "clear_tags_now"

# Handle clear flag safely
if st.session_state.get(tags_clear_flag):
    st.session_state[tags_key] = []
    st.session_state[new_tag_key] = ""
    st.session_state[tags_clear_flag] = False
elif tags_key not in st.session_state:
    # Only initialize once with prediction
    st.session_state[tags_key] = predicted_tags

if new_tag_key not in st.session_state:
    st.session_state[new_tag_key] = ""

tag_options = sorted(list(st.session_state.tag_mlb.classes_))

# --- Display the UI ---
tags_input_col, clear_tags_col = st.columns([5, 1])
with tags_input_col:
    # Use the 'key' argument to directly bind the widget's state
    tags_input = st.multiselect(
        "Select, add, or remove tags:",
        options=tag_options,
        key=tags_key  # Bind to st.session_state.tags_input
    )
    new_tag_input = st.text_input(
        "Add new tags (comma-separated):",
        key=new_tag_key # Bind to st.session_state.new_tag_input
    )
with clear_tags_col:
    if st.button("🗙", key="clear_tags_button"):
        st.session_state[tags_clear_flag] = True
        st.rerun()

# --- Merge old + new tags, dedupe, sort ---
new_tags_list = [t.strip() for t in st.session_state[new_tag_key].split(",") if t.strip()]
final_tags_list = sorted(list(set(st.session_state[tags_key] + new_tags_list)))

# --- Confirm & Save Button ---
if st.button("Confirm & Save", use_container_width=True):
    # Save the row for retraining
    sorted_row = {
        'note': current_note,
        'tags': ",".join(final_tags_list),
        'folder': final_folder,
        'current_folder': current_folder,
        'ai_suggested_folder': ai_suggested_folder,
        'url': url
    }
    pd.DataFrame([sorted_row]).to_csv(
        NEWLY_SORTED_DATA_FILE,
        mode='a',
        header=not os.path.exists(NEWLY_SORTED_DATA_FILE),
        index=False
    )

    # ✅ Remove row from full df_full not just filtered df!
    full_index = df_full.index[(df_full['url'] == url)].tolist()
    if full_index:
        df_remaining = df_full.drop(full_index).reset_index(drop=True)
    else:
        df_remaining = df_full


    df_remaining.to_csv(REMAINING_BOOKMARKS_FILE, index=False, sep=';')
    st.session_state['df_full'] = df_remaining
    push_to_github(NEWLY_SORTED_DATA_FILE, NEWLY_SORTED_DATA_FILE)
    push_to_github(REMAINING_BOOKMARKS_FILE, REMAINING_BOOKMARKS_FILE)


    # --- Model Update Logic ---
    is_new_folder = final_folder not in st.session_state.folder_encoder.classes_
    added_new_tag = any(tag not in st.session_state.tag_mlb.classes_ for tag in final_tags_list)

    if is_new_folder or added_new_tag:
        st.session_state.new_items_staged_for_retrain += 1
        if is_new_folder:
            st.warning(f"Saved new folder '{final_folder}'. Retrain later to teach the AI.")
        if added_new_tag:
            st.warning(f"Saved new tag(s). Retrain later to teach the AI.")
    else:
        st.info("✅ Saved! (No partial fit since the folder model is calibrated. Do full retrain later.)")

    st.success(f"✅ Saved and removed from remaining.")
    # This block is CRITICAL. It clears the state for the next item.
    for key in [note_key, folder_key, tags_key, new_folder_key, new_tag_key]:
        if key in st.session_state:
            del st.session_state[key]
                # Clear the cached tag predictions so they're recalculated for the next item
    if 'predicted_tags' in st.session_state:
        del st.session_state['predicted_tags']
    st.rerun()

if st.button("Delete", use_container_width=True):
    # Override values
    broken_row = {
        'note': "broken link",
        'tags': "delete",
        'folder': "delete",
        'current_folder': current_folder,
        'ai_suggested_folder': ai_suggested_folder,
        'url': url
    }

    pd.DataFrame([broken_row]).to_csv(
        NEWLY_SORTED_DATA_FILE,
        mode='a',
        header=not os.path.exists(NEWLY_SORTED_DATA_FILE),
        index=False
    )

    # Remove from remaining
    full_index = df_full.index[(df_full['url'] == url)].tolist()
    if full_index:
        df_remaining = df_full.drop(full_index).reset_index(drop=True)
    else:
        df_remaining = df_full

    df_remaining.to_csv(REMAINING_BOOKMARKS_FILE, index=False, sep=';')
    st.session_state['df_full'] = df_remaining

    st.info("🗑️ Marked as broken link and scheduled for deletion.")

    # Reset state
    for key in [note_key, folder_key, tags_key, new_folder_key, new_tag_key]:
        if key in st.session_state:
            del st.session_state[key]
                # Clear the cached tag predictions so they're recalculated for the next item
    if 'predicted_tags' in st.session_state:
        del st.session_state['predicted_tags']
    st.rerun()


# --- Additional Options ---
st.divider()

if st.button("Show Top 5 Folder Suggestions", use_container_width=True):
    try:
        probs = st.session_state.folder_model.predict_proba(X_folder_vec)[0]
        suggestions = sorted(zip(probs, st.session_state.folder_encoder.classes_), reverse=True)
        st.caption("Top 5 Folder Suggestions:")
        for prob, folder in suggestions[:5]:
            st.write(f"{prob*100:5.1f}% → {folder}")
    except (AttributeError, ValueError):
        st.warning("Model isn't trained enough for probabilities.")

if st.button("Show All Folders", use_container_width=True):
    st.caption("All Available Folders:")
    all_folders = sorted(list(st.session_state.folder_encoder.classes_))
    st.text("\n".join(all_folders))
    
if st.button("Show All Tags", use_container_width=True):
    st.caption("All Available Tags:")
    all_tags = sorted(list(st.session_state.tag_mlb.classes_))
    st.text("\n".join(all_tags))

st.divider()
retrain_button_label = f"🔁 Retrain Models ({st.session_state.new_items_staged_for_retrain} new items staged)"
if st.button(retrain_button_label, use_container_width=True):
    full_retrain_all_models(st.session_state.embedding_model)
    st.session_state.new_items_staged_for_retrain = 0

