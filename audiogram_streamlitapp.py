import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load structured dataset
# -------------------------------
audiogram_df = pd.read_csv("audiogram_df.csv")

# -------------------------------
# App Title
# -------------------------------
st.title("🎧 Reg-ent AI Audiogram Demo")

# -------------------------------
# Section 1: Unstructured Data
# -------------------------------
st.header("Unstructured Data: Audiograms")
st.markdown("""
Audiograms are typically considered **unstructured data** in EHRs because they are most often stored as scanned images or PDFs rather than as discrete values.  
This creates inconsistency, as many providers do not enter structured audiometry data directly into the EHR but instead upload scanned copies.  

As a result, extracting audiogram information has historically been difficult and, in many cases, not possible.  
While standards such as **LOINC** and **SNOMED** exist for hearing thresholds, most EHRs do not consistently apply them,  
further limiting standardization and interoperability.
""")

st.image("sample_audiogram.png", caption="Example of Unstructured Audiogram (Scanned Image)", use_container_width=True)


# -------------------------------
# Section 2: Structured Dataset
# -------------------------------
st.header("Structured Audiogram Dataset")
st.markdown("""
The **Reg-ent Registry** is implementing AI-driven **natural language modeling (NLM)** processes 
to extract results directly from scanned audiograms.  

This innovation will allow us to **convert audiogram data into structured fields**, 
something that was previously not possible. By making audiograms extractable and standardized, 
Reg-ent can **expand research opportunities** and support the development of new quality measures.
""")

# Show the synthesized dataset directly under the text
st.dataframe(audiogram_df, use_container_width=True, hide_index=True)

# -------------------------------
# Section 3: Interactive Exploration
# -------------------------------
st.header("Interactive Exploration")
category = st.sidebar.multiselect(
    "Filter by Hearing Category:",
    options=audiogram_df["Category"].unique(),
    default=audiogram_df["Category"].unique()
)

filtered_df = audiogram_df[audiogram_df["Category"].isin(category)]
st.write("Filtered view of dataset based on hearing category:")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# Show summary statistics by category
st.subheader("📈 Summary Statistics by Hearing Category")
summary_df = filtered_df.groupby("Category")[["PTA_Right", "PTA_Left", "WRS_Right", "WRS_Left"]].mean().round(1).reset_index()
st.dataframe(summary_df, use_container_width=True, hide_index=True)


# -------------------------------
# Section 3b: Grouped Bar Chart
# -------------------------------
st.subheader("📊 Grouped Bar Chart: PTA, SRT, SDT by Hearing Loss Type")

# Collapse to ear-level dataset
ear_data = []
for side in ["Right", "Left"]:
    tmp = audiogram_df[["PatientID", f"PTA_{side}", f"SRT_{side}", f"SDT_{side}", "Category"]].copy()
    tmp = tmp.rename(columns={
        f"PTA_{side}": "PTA",
        f"SRT_{side}": "SRT",
        f"SDT_{side}": "SDT"
    })
    tmp["Ear"] = side
    ear_data.append(tmp)

ear_df = pd.concat(ear_data)

# Compute mean PTA, SRT, SDT by Category
cat_means = ear_df.groupby("Category")[["PTA", "SRT", "SDT"]].mean().reset_index()

# Grouped bar chart
fig, ax = plt.subplots(figsize=(8,6))
cat_means.plot(x="Category", kind="bar", ax=ax, rot=45)

ax.set_ylabel("dB HL")
ax.set_title("Mean PTA, SRT, and SDT by Category")
ax.legend(title="Metric")
st.pyplot(fig)


# -------------------------------
# Section 3e: Heatmap
# -------------------------------
st.subheader("🔥 Heatmap: PTA, SRT, SDT by Category")

import seaborn as sns

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(
    cat_means.set_index("Category"),
    annot=True, cmap="coolwarm", fmt=".1f", cbar_kws={'label': 'dB HL'},
    ax=ax
)

ax.set_title("Heatmap of Mean PTA, SRT, SDT by Category")
st.pyplot(fig)












