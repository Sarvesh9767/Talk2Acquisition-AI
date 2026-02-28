import streamlit as st
from search_engine import MetadataSearchEngine

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Talk2Acquisition AI", layout="wide")

st.title("💬 Talk2Acquisition - AI Metadata Assistant")

# ---------------------------
# Load Search Engine (Cached)
# ---------------------------
@st.cache_resource
def load_engine():
    return MetadataSearchEngine("talk2acquisition_master_metadata_v2.csv")

engine = load_engine()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("🔎 Filters")

selected_vendor = st.sidebar.selectbox(
    "Vendor",
    ["All"] + sorted(engine.df["vendor"].unique().tolist())
)

selected_asset = st.sidebar.selectbox(
    "Asset Class",
    ["All"] + sorted(engine.df["asset_class"].unique().tolist())
)

# ---------------------------
# Chat History State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# User Input
# ---------------------------
query = st.chat_input("Ask about any data attribute...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            results = engine.search(query)

            # Apply sidebar filters
            if selected_vendor != "All":
                results = [r for r in results if r["vendor"] == selected_vendor]

            if selected_asset != "All":
                results = [r for r in results if r["asset_class"] == selected_asset]

            if not results:
                response = "❌ No strong match found."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Render structured output
                for i, r in enumerate(results[:5], 1):
                    st.markdown(f"### #{i} 📌 {r['attribute_name']}")

                    st.markdown(f"**🏢 Vendor:** {r['vendor']}")
                    st.markdown(f"**📊 Asset Class:** {r['asset_class']}")
                    st.markdown(f"**📝 Definition:** {r['definition']}")
                    st.markdown(f"**🔄 Frequency:** {r['frequency']}")
                    st.markdown(f"**👤 Business Owner:** {r['business_owner']}")
                    st.markdown(f"**🧑‍💼 Data Steward:** {r['data_steward']}")
                    st.markdown(f"**🏛 Regulatory Source:** {r['regulatory_source']}")
                    st.markdown(
                        f"**⭐ Confidence:** {r['confidence']}  |  📊 Similarity: {r['similarity']}%"
                    )

                    st.divider()

                # Store plain text summary in chat history (optional minimal log)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Returned top {min(len(results),5)} results."
                })
