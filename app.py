import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Talk2Acquisition", layout="wide")
st.title("Talk2Acquisition – AI Acquisition Intelligence Engine")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("talk2acquisition_master_metadata_v2.csv")
    df.fillna("")
    return df

df = load_data()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------------------------
# PREP SEARCH TEXT
# -------------------------------------------------
df["search_text"] = (
    df["attribute_name"].astype(str) + " " +
    df["definition"].astype(str) + " " +
    df["synonyms"].astype(str) + " " +
    df["vendor"].astype(str) + " " +
    df["asset_class"].astype(str)
)

@st.cache_data
def generate_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

embeddings = generate_embeddings(df["search_text"].tolist())

# -------------------------------------------------
# SEARCH
# -------------------------------------------------
def search(query):
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, embeddings)[0]
    temp_df = df.copy()
    temp_df["similarity"] = similarity
    return temp_df.sort_values("similarity", ascending=False).head(25)

# -------------------------------------------------
# ATTRIBUTE DETECTION
# -------------------------------------------------
def detect_best_attribute(results, query):
    grouped = results.groupby("attribute_name")
    attribute_scores = []
    query_lower = query.lower()

    for attribute, group in grouped:
        avg_sim = group["similarity"].mean()
        max_sim = group["similarity"].max()
        count_weight = len(group) / 25

        attribute_score = 0.5*avg_sim + 0.3*max_sim + 0.2*count_weight

        if attribute.lower() in query_lower:
            attribute_score += 0.20

        synonyms_text = " ".join(group["synonyms"].astype(str)).lower()
        if any(word in synonyms_text for word in query_lower.split()):
            attribute_score += 0.10

        attribute_scores.append({
            "attribute_name": attribute,
            "attribute_score": attribute_score
        })

    score_df = pd.DataFrame(attribute_scores)
    score_df = score_df.sort_values("attribute_score", ascending=False)
    return score_df.iloc[0]

# -------------------------------------------------
# VENDOR SCORING LOGIC
# -------------------------------------------------
def vendor_score(row):

    score = 0

    # Active weight
    if row["is_active"] == "Yes":
        score += 40

    # Frequency weight
    freq_weight = {
        "Real-time": 25,
        "Intraday": 20,
        "Daily": 15,
        "Weekly": 10,
        "Monthly": 5,
        "Static": 2
    }
    score += freq_weight.get(row["frequency"], 0)

    # Regulatory weight
    if row["regulatory_source"] in ["ISO","SEC","ESMA","FCA","Basel","MiFID","IFRS"]:
        score += 20

    # Recency weight
    try:
        last_updated = pd.to_datetime(row["last_updated"])
        days_old = (datetime.now() - last_updated).days
        if days_old < 30:
            score += 15
        elif days_old < 90:
            score += 10
        else:
            score += 5
    except:
        pass

    return score

# -------------------------------------------------
# MAIN QUERY
# -------------------------------------------------
query = st.text_input("Ask a metadata or governance question")

if query:

    results = search(query)
    best_attribute = detect_best_attribute(results, query)

    top_attribute = best_attribute["attribute_name"]
    confidence_score = best_attribute["attribute_score"]

    if confidence_score < 0.32:
        st.warning("Attribute not confidently identified. Initiate onboarding.")

    else:
        st.subheader(f"Attribute Identified: {top_attribute}")
        st.write(f"Detection Confidence: {round(confidence_score*100,2)}%")

        grouped = df[df["attribute_name"] == top_attribute].copy()

        # Apply scoring
        grouped["vendor_score"] = grouped.apply(vendor_score, axis=1)

        # Sort by score
        grouped = grouped.sort_values("vendor_score", ascending=False)

        st.markdown("### Available Vendors (AI Ranked)")
        st.dataframe(grouped.drop(columns=["search_text"]),
                     use_container_width=True)

        # Recommended vendor
        recommended_vendor = grouped.iloc[0]["vendor"]
        st.success(f"⭐ Recommended Vendor: {recommended_vendor}")

        # Vendor selection
        selected_vendor = st.selectbox(
            "Choose vendor (override allowed):",
            grouped["vendor"].unique()
        )

        selected_row = grouped[grouped["vendor"] == selected_vendor].iloc[0]

        # -------------------------------------------------
        # FULL METADATA DISPLAY
        # -------------------------------------------------
        st.markdown("### Selected Vendor Details")

        col1, col2 = st.columns(2)
        columns = list(selected_row.index)

        if "search_text" in columns:
            columns.remove("search_text")

        half = len(columns)//2

        with col1:
            for col in columns[:half]:
                st.write(f"**{col}** : {selected_row[col]}")

        with col2:
            for col in columns[half:]:
                st.write(f"**{col}** : {selected_row[col]}")

        # -------------------------------------------------
        # DOWNLOAD BUTTON
        # -------------------------------------------------
        download_df = selected_row.to_frame().T
        csv = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Selected Metadata",
            data=csv,
            file_name=f"{top_attribute}_{selected_vendor}_metadata.csv",
            mime="text/csv"
        )

        # Confirm
        if st.button("Confirm Consumption"):
            st.success("Consumption request captured (POC mode).")
