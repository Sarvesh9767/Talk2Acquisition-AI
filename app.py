import streamlit as st
from search_engine import MetadataSearchEngine
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AskMetadata",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 AskMetadata – AI Metadata Assistant")
st.write("Ask governance or metadata-related questions.")

st.markdown(
"""
Examples:

• Is ISIN active?  
• Who owns coupon rate?  
• What is update frequency of equity price?  
• Regulatory source for bond yield?  
"""
)

# -----------------------------
# LOAD ENGINE
# -----------------------------
@st.cache_resource
def load_engine():
    return MetadataSearchEngine("talk2acquisition_master_metadata_v2.csv")

engine = load_engine()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

vendor_filter = st.sidebar.selectbox(
    "Vendor",
    ["All", "Bloomberg", "Refinitiv", "SIX", "Internal"]
)

asset_filter = st.sidebar.selectbox(
    "Asset Class",
    ["All", "Equity", "Bond", "FX", "All"]
)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# USER INPUT
# -----------------------------
query = st.chat_input("Ask about any data attribute")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching metadata..."):

            results, detected_filters = engine.search(query)

            # Apply manual filters
            if vendor_filter != "All":
                results = results[results["vendor"] == vendor_filter]

            if asset_filter != "All":
                results = results[results["asset_class"] == asset_filter]

            if len(results) == 0:
                response = "Attribute not confidently identified. Initiate onboarding request."
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            else:

                full_response = ""

                st.subheader("🔎 AI Search Results")

                if detected_filters.get("asset_class"):
                    st.info(f"🤖 AI detected Asset Class: {detected_filters['asset_class']}")

                for i, row in results.iterrows():

                    similarity_pct = round(row["similarity"] * 100, 2)

                    if similarity_pct > 75:
                        confidence = "🟢 High"
                    elif similarity_pct > 55:
                        confidence = "🟡 Moderate"
                    else:
                        confidence = "🔴 Low"

                    block = f"""
# 📌 {row['attribute_name']}

🏢 Vendor: {row['vendor']}

📊 Asset Class: {row['asset_class']}

📝 Definition: {row['definition']}

🔄 Frequency: {row['frequency']}

👤 Business Owner: {row['business_owner']}

🧑‍💼 Data Steward: {row['data_steward']}

🏛 Regulatory Source: {row['regulatory_source']}

⭐ Confidence: {confidence} | 📊 Similarity: {similarity_pct}%
"""

                    st.markdown(block)

                    full_response += block

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
