# Agentic Healthcare Assistant Capstone

This repository contains a simple deployment-ready version of the **Agentic Healthcare Assistant for Medical Task Automation** capstone. The build keeps the architecture straightforward while still demonstrating the core assignment requirements:

- patient lookup from a registry (`records.xlsx`)
- medical-history retrieval from PDF reports
- mock appointment booking
- medical information summarization
- lightweight patient memory
- Streamlit dashboard with logs and evaluation metrics

## Project Structure

```text
healthcare_capstone_project/
├── app.py
├── healthcare_agent.py
├── requirements.txt
├── README.md
└── data/
    ├── records.xlsx
    ├── sample_patient.pdf
    ├── sample_report_anjali.pdf
    ├── sample_report_david.pdf
    └── sample_report_ramesh.pdf
```

## 1. Local Setup

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key before running:

```bash
export OPENAI_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

## 2. Run the Streamlit App

```bash
streamlit run app.py
```

## 3. Suggested Demo Prompts

- `What is Anjali Mehra's diagnosis and treatment plan?`
- `Book a nephrologist for David Thompson and summarize diabetes care.`
- `Show Ramesh Kulkarni's history and explain hypertension follow-up.`

## 4. Notes for Streamlit Cloud / GitHub

- Upload the entire project folder to GitHub.
- Make sure `app.py` is in the repository root.
- Add `OPENAI_API_KEY` in your Streamlit app secrets.
- Do **not** include `!pip install` lines in any deployed Python file.

## 5. Important Demo Limitation

This is an **educational capstone demo**. It uses local files and a mock scheduling database. It is not connected to real EHR systems, hospital APIs, or live medical websites.
