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
# Section 3b: Radar Graph Explorer
# -------------------------------
st.subheader("🎯 Radar Graph: PTA vs SRT vs SDT")

# Patient selection
patient_id = st.selectbox("Select Patient for Radar Graph:", audiogram_df["PatientID"].unique())
patient_data = audiogram_df[audiogram_df["PatientID"] == patient_id].iloc[0]

# Axes = PTA, SRT, SDT for both ears
categories = ["PTA_Right", "PTA_Left", "SRT_Right", "SRT_Left", "SDT_Right", "SDT_Left"]

pta_values = [patient_data["PTA_Right"], patient_data["PTA_Left"],
              None, None, None, None]
srt_values = [None, None,
              patient_data["SRT_Right"], patient_data["SRT_Left"],
              None, None]
sdt_values = [None, None, None, None,
              patient_data["SDT_Right"], patient_data["SDT_Left"]]

# Fill in radar data (keeping each metric isolated)
pta_plot = [patient_data["PTA_Right"], patient_data["PTA_Left"], 0, 0, 0, 0]
srt_plot = [0, 0, patient_data["SRT_Right"], patient_data["SRT_Left"], 0, 0]
sdt_plot = [0, 0, 0, 0, patient_data["SDT_Right"], patient_data["SDT_Left"]]

# Close radar loops
pta_plot += pta_plot[:1]
srt_plot += srt_plot[:1]
sdt_plot += sdt_plot[:1]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Plot radar chart
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

ax.plot(angles, pta_plot, linewidth=2, color="orange", label="PTA")
ax.fill(angles, pta_plot, alpha=0.25, color="orange")

ax.plot(angles, srt_plot, linewidth=2, color="blue", label="SRT")
ax.fill(angles, srt_plot, alpha=0.25, color="blue")

ax.plot(angles, sdt_plot, linewidth=2, color="green", label="SDT")
ax.fill(angles, sdt_plot, alpha=0.25, color="green")

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Radial axis (demo scale 10–60, can change to 0–110 for full clinical)
ax.set_rlabel_position(0)
ax.set_yticks([10, 20, 30, 40, 50, 60])
ax.set_yticklabels(["10","20","30","40","50","60"])
ax.set_ylim(10, 60)

ax.set_title(f"Patient {patient_id} - PTA vs SRT vs SDT")
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

st.pyplot(fig)





