import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AskMetadata",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AskMetadata – AI Metadata Discovery Assistant")
st.caption("AI-powered discovery for vendor metadata, attributes, and governance information.")

st.markdown("""
💡 **Example Questions**

• Where is ISIN available?  
• Which vendor provides Coupon Rate?  
• Who owns Equity Price?  
• What is the update frequency of Bond Yield?  
• Show vendors providing Ticker
""")

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("talk2acquisition_master_metadata_v2.csv")
    df.fillna("", inplace=True)
    return df

df = load_data()

# -----------------------------
# Load AI Model
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Prepare Embeddings
# -----------------------------

@st.cache_resource
def create_embeddings(df):

    search_text = (
        df["attribute_name"] + " " +
        df["definition"] + " " +
        df["synonyms"]
    )

    embeddings = model.encode(search_text.tolist(), convert_to_tensor=True)

    return embeddings

embeddings = create_embeddings(df)

# -----------------------------
# Vendor Scoring Logic
# -----------------------------

def score_vendor(row):

    score = 0

    if str(row["is_active"]).lower() == "yes":
        score += 3

    if row["frequency"] in ["Real-time", "Daily"]:
        score += 2

    if row["regulatory_source"] != "":
        score += 2

    if row["business_owner"] != "":
        score += 1

    return score


# -----------------------------
# Search Engine
# -----------------------------

def search_metadata(query, top_k=5):

    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, embeddings)[0]

    top_results = scores.argsort(descending=True)[:top_k]

    results = []

    for idx in top_results:

        row = df.iloc[int(idx)]

        similarity = float(scores[int(idx)])

        if similarity > 0.70:
            confidence = "High"
        elif similarity > 0.50:
            confidence = "Moderate"
        else:
            confidence = "Low"

        vendor_score = score_vendor(row)

        results.append({
            "attribute_id": row["attribute_id"],
            "attribute_name": row["attribute_name"],
            "vendor": row["vendor"],
            "file_name": row["file_name"],
            "asset_class": row["asset_class"],
            "definition": row["definition"],
            "data_type": row["data_type"],
            "frequency": row["frequency"],
            "synonyms": row["synonyms"],
            "sample_values": row["sample_values"],
            "regulatory_source": row["regulatory_source"],
            "country_coverage": row["country_coverage"],
            "business_owner": row["business_owner"],
            "data_steward": row["data_steward"],
            "is_active": row["is_active"],
            "last_updated": row["last_updated"],
            "confidence": confidence,
            "similarity": round(similarity * 100, 2),
            "vendor_score": vendor_score
        })

    return results


# -----------------------------
# Chat UI
# -----------------------------

query = st.chat_input("Ask metadata question...")

if query:

    st.chat_message("user").write(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching metadata..."):

            results = search_metadata(query)

            if len(results) == 0:
                st.warning("Attribute not confidently identified. Initiate onboarding request.")
            else:

                for i, r in enumerate(results):

                    st.markdown(f"""
### #{i+1} 📌 {r['attribute_name']}

🏢 **Vendor:** {r['vendor']}  
📁 **File Name:** {r['file_name']}  
📊 **Asset Class:** {r['asset_class']}  

📝 **Definition:**  
{r['definition']}

📦 **Data Type:** {r['data_type']}  
🔄 **Frequency:** {r['frequency']}  
🔗 **Synonyms:** {r['synonyms']}  
📊 **Sample Values:** {r['sample_values']}

🏛 **Regulatory Source:** {r['regulatory_source']}  
🌍 **Country Coverage:** {r['country_coverage']}

👤 **Business Owner:** {r['business_owner']}  
🧑‍💼 **Data Steward:** {r['data_steward']}

✅ **Is Active:** {r['is_active']}  
🕒 **Last Updated:** {r['last_updated']}

⭐ **Confidence:** {r['confidence']}  
📊 **Similarity:** {r['similarity']}%  

🏆 **Vendor Score:** {r['vendor_score']}
""")

                # -----------------------------
                # Download Button
                # -----------------------------

                download_df = pd.DataFrame(results)

                st.download_button(
                    "⬇ Download Selected Metadata",
                    download_df.to_csv(index=False),
                    file_name="selected_metadata.csv",
                    mime="text/csv"
                )
