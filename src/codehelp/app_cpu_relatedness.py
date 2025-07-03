import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity ,  euclidean_distances, manhattan_distances
# Load models
import joblib
from sentence_transformers import SentenceTransformer
import torch
@st.cache_resource
def load_models():
    xgb_model = joblib.load("G:\Viren-minor-project\Viren-minor-project\check_relatedness_temp\error_question_classifier_xgb_with_3_dataset.pkl")
    model = SentenceTransformer("G:/Viren-minor-project/Viren-minor-project/check_relatedness_temp/sbert_model_dir",device='cpu')# or load from pickle if custom
    return xgb_model, model

xgb_model, model = load_models()

# sentence_model._target_device = 'cpu'
def extract_features(error_emb, question_emb):
    """Compute multiple similarity metrics between two embeddings."""
    cos_sim = cosine_similarity([error_emb], [question_emb])[0][0]
    euc_dist = euclidean_distances([error_emb], [question_emb])[0][0]
    man_dist = manhattan_distances([error_emb], [question_emb])[0][0]
    return [cos_sim, euc_dist, man_dist]
# Streamlit UI
st.title("🧠 Virtual TA - Relatedness Checker")
st.markdown("Check if the Python error and user issue are related.")

# Inputs
error_msg = st.text_area("🔴 Python Error Message", height=100)
issue_description = st.text_area("🟡 User Question / Issue", height=100)
# optional_context = st.text_area("🟢 Additional Comments (Optional)", height=50)

# Button
if st.button("Check Relatedness"):
    if error_msg.strip() and issue_description.strip():
        # Combine error and issue (or pass them separately if needed)
        error_vec = model.encode(error_msg,convert_to_numpy=True)
        issue_vec = model.encode(issue_description,convert_to_numpy=True)

        # Combine embeddings — method may vary
        # error_emb = model.encode(error_text, convert_to_numpy=True)
    # question_emb = model.encode(question_text, convert_to_numpy=True)

        features = extract_features(error_vec, issue_vec)
        prediction = xgb_model.predict([features])[0]  # shape (2*768,) if using MiniLM

        # Predict
        pred = xgb_model.predict([features])[0]
        label = "✅ Related" if pred == 1 else "❌ Not Related"

        st.subheader("Prediction:")
        st.success(label if pred == 1 else label)

    else:
        st.warning("Please enter both the error and the issue.")

