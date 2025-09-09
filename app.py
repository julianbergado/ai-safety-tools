# Imports and Model Loading

import streamlit as st
from transformers import pipeline
import pandas as pd

print("Loading classification model...")
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli")
print("Model loaded successfully!")


# Policies from the EU Artificial Intelligence Act
policy_texts = [
    # --- Prohibited AI Practices ---
    "Employing subliminal, manipulative, or deceptive techniques to distort behavior and cause significant harm.",
    "Exploiting vulnerabilities related to age, disability, or a person's social or economic situation to cause significant harm.",
    "Using biometric categorization systems to infer sensitive information such as race, political opinions, trade union membership, religious beliefs, or sexual orientation.",
    "Evaluating or classifying individuals or groups based on their social behavior or personal traits, leading to detrimental treatment in unrelated contexts (i.e., social scoring).",
    "Using 'real-time' remote biometric identification in public spaces for law enforcement, with limited exceptions for serious crimes or threats.",
    "Assessing the risk of an individual committing a crime based solely on profiling or personality traits.",
    "Creating or expanding facial recognition databases by untargeted scraping of images from the internet or CCTV footage.",
    "Inferring emotions in workplaces or educational settings, unless for medical or safety reasons.",

    # --- Requirements for High-Risk AI Systems ---
    "Providers of high-risk AI systems must conduct a conformity assessment procedure before their products can be sold and used in the EU.",
    "High-risk AI systems must comply with a range of requirements including for testing, data training and cybersecurity.",
    "Providers of high-risk AI systems will in some cases have to conduct a fundamental rights impact assessment to ensure their systems comply with EU law.",
    "After high-risk AI systems are placed in the market, providers must implement post-market monitoring and take corrective actions if necessary.",

    # --- Transparency Obligations ---
    "Users must be made aware that they are interacting with an AI system, such as a chatbot.",
    "Deployers of AI systems that generate or manipulate image, audio or video content (i.e. deep fakes), must disclose that the content has been artificially generated or manipulated.",
    "Providers of AI systems that generate large quantities of synthetic content must implement reliable techniques like watermarks to identify the output as AI-generated.",
    "Employers who deploy AI systems in the workplace must inform the workers and their representatives.",

    # --- Rules for General-Purpose AI (GPAI) ---
    "All GPAI models will have to draw up and maintain up-to-date technical documentation.",
    "All providers of GPAI models have to put a policy in place to respect Union copyright law.",
    "GPAI models must draw up and make publicly available a sufficiently detailed summary of the content used for training.",
    "Providers of systemic-risk GPAI models are required to constantly assess and mitigate the risks they pose and to ensure cybersecurity protection.",    
]

candidate_labels = [
    'Risk Management', 
    'Data Privacy & Security', 
    'Transparency & Explainability', 
    'Public Safety', 
    'Fairness & Non-Discrimination', 
    'Prohibited AI Practices'
]


# UI

# Using Streamlit's caching to load the model only once
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# --- Streamlit App Interface ---
st.title("AI Policy Clause Classifier 🤖")
st.write(
    "Select a clause from an AI policy document to see its thematic classification. "
    "This demo uses a zero-shot classification model to categorize text without prior specific training."
)

# A dropdown menu for the user to select a policy clause
selected_clause = st.selectbox("Choose a policy clause to classify:", policy_texts)

# A button to trigger the classification
if st.button("Classify Clause"):
    if selected_clause:
        with st.spinner('Analyzing the text...'):
            # Run the classification
            result = classifier(selected_clause, candidate_labels, multi_label=False)
            
            st.subheader("Classification Results:")
            
            # Neatly display labels and scores
            for label, score in zip(result['labels'], result['scores']):
                st.write(f"- **{label}:** {score:.2%}")

            # Bonus: Display a bar chart for better visualization
            st.subheader("Visualized Scores")
            chart_data = pd.DataFrame({'Score': result['scores']}, index=result['labels'])
            st.bar_chart(chart_data)
    else:
        st.warning("Please select a clause first.")
