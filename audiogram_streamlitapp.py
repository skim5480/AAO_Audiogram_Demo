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

# Show a PNG example of an unstructured audiogram
st.image("sample_audiogram.png", caption="Example of Unstructured Audiogram (Scanned Image)", use_column_width=True)

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
st.dataframe(audiogram_df, use_container_width=True)

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
st.dataframe(filtered_df, use_container_width=True)

# Show summary statistics by category
st.subheader("📈 Summary Statistics by Hearing Category")
st.write(filtered_df.groupby("Category")[["PTA_Right", "PTA_Left", "WRS_Right", "WRS_Left"]].mean().round(1))

# -------------------------------
# Section 3b: Radar Graph Explorer
# -------------------------------
st.subheader("🎯 Radar Graph: PTA & SRT by Patient")

# Patient selection
patient_id = st.selectbox("Select Patient for Radar Graph:", filtered_df["PatientID"].unique())
patient_data = filtered_df[filtered_df["PatientID"] == patient_id].iloc[0]

# Define categories and values (PTA + SRT for both ears)
categories = ["PTA_Right", "PTA_Left", "SRT_Right", "SRT_Left"]
values = [patient_data[c] for c in categories]

# Close radar loop
values += values[:1]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Plot radar chart
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Patient {patient_id}")
ax.fill(angles, values, alpha=0.25)

# Category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Radial axis (0–110 dB HL typical)
ax.set_rlabel_position(0)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20","40","60","80","100"])
ax.set_ylim(0, 110)

ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
st.pyplot(fig)

