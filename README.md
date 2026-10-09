# AI-Powered Call Analysis & Admin Portal

## Overview

A Python-based web application that analyzes sample customer calls and presents the results through an admin dashboard.

The application provides audio playback, sentiment analysis, purchase-likelihood estimates, customer insights, recommended follow-up actions, and downloadable PDF reports.

## Features

- Four sample customer call recordings
- Audio playback and call transcripts
- AI-based sentiment classification: Positive, Neutral, and Negative
- Logistic Regression model for experimental purchase-likelihood estimates
- Customer signal extraction and detailed insights
- Recommended next actions for each call
- PDF report generation and download
- Streamlit dashboard and FastAPI backend

## Technologies Used

- Python
- Streamlit
- FastAPI
- Scikit-learn
- Hugging Face Transformers
- Pandas and NumPy
- ReportLab

## Project Structure

call-analysis-portal/
├── app/
│   ├── __init__.py
│   ├── analysis.py
│   ├── data.py
│   ├── main.py
│   └── static/
│       └── audio/
├── scripts/
│   └── generate_audio.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore


## Setup and Run

### 1. Clone the repository

bash:
git clone YOUR_GITHUB_REPOSITORY_URL
cd call-analysis-portal


### 2. Create a virtual environment

Windows PowerShell:

powershell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1


### 3. Install dependencies

powershell:
python -m pip install -r requirements.txt


### 4. Start the FastAPI backend

In the first terminal:

powershell:
python -m uvicorn app.main:app --reload


API documentation: http://127.0.0.1:8000/docs

### 5. Start the Streamlit dashboard

In a second terminal, activate the same virtual environment and run:

powershell:
streamlit run streamlit_app.py


Open http://localhost:8501 in your browser.

## Machine Learning Notes

A pretrained Transformer model is used for sentiment classification. Logistic Regression estimates purchase likelihood from extracted customer signals.

The current purchase model uses synthetic demonstration training examples. Its output is experimental and has not been validated or calibrated against real customer purchase outcomes. The displayed percentages should not be interpreted as reliable real-world probabilities.

The included calls are sample interactions intended for demonstration.

## Future Improvements

- Train and validate the purchase model on labeled historical customer interactions.
- Evaluate model performance and probability calibration.
- Improve negation and context detection.
- Add secure administrator authentication.
- Support uploaded call recordings and automatic speech transcription.

## Disclaimer

This project is an educational demonstration and is not intended for making real-world sales decisions without further validation.
