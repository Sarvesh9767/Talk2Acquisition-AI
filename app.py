import streamlit as st
from search_engine import MetadataSearchEngine
import pandas as pd

st.set_page_config(page_title="AskMetadata", layout="wide")

st.title("📊 AskMetadata – AI Metadata Assistant")
st.write("Ask governance or metadata related questions.")

st.write(
"""
Examples:

• Is ISIN active?  
• Who owns coupon rate?  
• What is update frequency of equity price?  
• Regulatory source for bond yield?
"""
)

@st.cache_resource
def load_engine():
    return MetadataSearchEngine()

engine = load_engine()

query = st.chat_input("Ask about metadata attributes...")

if query:

    st.chat_message("user").write(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching metadata..."):

            results = engine.search(query)

        if len(results) == 0:
            st.warning("No metadata found.")
        else:

            st.subheader("🔎 AI Search Results")

            for i, row in results.iterrows():

                similarity_pct = round(row["similarity"] * 100, 2)

                st.markdown(
f"""
### #{i+1} 📌 {row['attribute_name']}

🏢 **Vendor:** {row['vendor']}

📊 **Asset Class:** {row['asset_class']}

📝 **Definition:** {row['definition']}

🔄 **Frequency:** {row['frequency']}

👤 **Business Owner:** {row['business_owner']}

🧑‍💼 **Data Steward:** {row['data_steward']}

🏛 **Regulatory Source:** {row['regulatory_source']}

⭐ **Confidence:** {row['confidence']} | 📊 **Similarity:** {similarity_pct}%
"""
                )

            st.divider()

            st.subheader("📋 Selected Vendor Details")

            selected_index = st.selectbox(
                "Choose vendor record",
                results.index,
                format_func=lambda x: f"{results.loc[x,'attribute_name']} ({results.loc[x,'vendor']})"
            )

            selected = results.loc[selected_index]

            detail_df = pd.DataFrame({
                "Field":[
                    "attribute_id",
                    "attribute_name",
                    "vendor",
                    "file_name",
                    "asset_class",
                    "definition",
                    "data_type",
                    "frequency",
                    "synonyms",
                    "sample_values",
                    "regulatory_source",
                    "country_coverage",
                    "business_owner",
                    "data_steward",
                    "is_active",
                    "last_updated"
                ],
                "Value":[
                    selected["attribute_id"],
                    selected["attribute_name"],
                    selected["vendor"],
                    selected["file_name"],
                    selected["asset_class"],
                    selected["definition"],
                    selected["data_type"],
                    selected["frequency"],
                    selected["synonyms"],
                    selected["sample_values"],
                    selected["regulatory_source"],
                    selected["country_coverage"],
                    selected["business_owner"],
                    selected["data_steward"],
                    selected["is_active"],
                    selected["last_updated"]
                ]
            })

            st.table(detail_df)

            st.download_button(
                label="⬇ Download Selected Metadata",
                data=detail_df.to_csv(index=False),
                file_name="selected_metadata.csv",
                mime="text/csv"
            )
