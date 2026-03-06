import streamlit as st
import pandas as pd
from search_engine import MetadataSearchEngine

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AskMetadata",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 AskMetadata – AI Metadata Assistant")
st.write("Ask governance or metadata-related questions.")

# -----------------------------
# Load Engine
# -----------------------------
@st.cache_resource
def load_engine():
    return MetadataSearchEngine("talk2acquisition_master_metadata_v2.csv")

engine = load_engine()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display Previous Messages
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
query = st.chat_input("Ask a metadata question...")

if query:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            results = engine.search(query)

            if len(results) == 0:
                response = "❌ No matching metadata found."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            else:

                response_text = ""

                for i, row in results.iterrows():

                    block = f"""
# 📌 {row['attribute_name']}

🏢 Vendor: {row['vendor']}

📊 Asset Class: {row['asset_class']}

📝 Definition: {row['definition']}

🔄 Frequency: {row['frequency']}

👤 Business Owner: {row['business_owner']}

🧑‍💼 Data Steward: {row['data_steward']}

🏛 Regulatory Source: {row['regulatory_source']}

⭐ Confidence: {row['confidence']} | 📊 Similarity: {row['similarity']:.2f}%
"""

                    st.markdown(block)

                    response_text += block

                st.session_state.messages.append({"role": "assistant", "content": response_text})
