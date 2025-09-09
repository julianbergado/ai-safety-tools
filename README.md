# AI Policy Clause Classifier

This project is a web-based tool designed to analyze and categorize clauses from AI policy and legal documents. Given a specific clause, the tool uses a zero-shot classification model to assign scores across several key governance themes, such as 'Risk Management' and 'Data Privacy'.

## How It Works

The application is built in Python and leverages several key technologies:

1.  **Core Model:** The classification is performed by `facebook/bart-large-mnli`, a powerful pre-trained model from the Hugging Face Hub. It's based on the PyTorch framework.
2.  **Zero-Shot Classification:** I chose a zero-shot approach because it allows for flexible and rapid analysis without the need for a large, manually-labeled dataset. This is ideal for an early-stage company that needs to quickly make sense of new policy documents. The model uses its general understanding of language to score the similarity between a policy clause and a set of candidate labels.
3.  **Interface:** The web application is built with Streamlit, which allowed for the creation of a clean, interactive user interface entirely in Python.

## Technical Stack

* **Python 3.10+**
* **Libraries:**
    * `transformers`: For accessing and using the Hugging Face model.
    * `torch`: The backend deep learning framework for the model.
    * `streamlit`: For building the interactive web application UI.
    * `pandas`: For data manipulation to display the results chart.

## How to Run This Project Locally

1.  Clone the repository:
    `git clone <your-repo-link>`
2.  Navigate to the project directory:
    `cd policy_classifier`
3.  Create and activate a virtual environment:
    `python -m venv venv && source venv/bin/activate`
4.  Install the required dependencies:
    `pip install -r requirements.txt`
5.  Run the Streamlit application:
    `streamlit run app.py`

The application will be available at `http://localhost:8501`.