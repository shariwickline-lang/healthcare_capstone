from __future__ import annotations

import streamlit as st

from healthcare_agent import HealthcareAssistant

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🩺", layout="wide")

st.title("🩺 Agentic Healthcare Assistant")
st.caption("Simple capstone deployment: patient lookup, history retrieval, appointment booking, medical info summaries, and memory.")


@st.cache_resource(show_spinner=True)
def load_assistant() -> HealthcareAssistant:
    return HealthcareAssistant()


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Project Overview")
    st.write(
        "This demo app wraps a simple agentic workflow around a patient registry, PDF-based retrieval, "
        "mock appointment booking, and a lightweight evaluation/logging layer."
    )
    st.info("Set your OPENAI_API_KEY in Streamlit secrets before running the app.")
    st.markdown(
        "**Suggested prompts**\n"
        "- What is Anjali Mehra's diagnosis and treatment plan?\n"
        "- Book a nephrologist for David Thompson and summarize diabetes care.\n"
        "- Show Ramesh Kulkarni's history and explain hypertension follow-up."
    )

try:
    assistant = load_assistant()
except Exception as exc:
    st.error(f"Setup error: {exc}")
    st.stop()

query_tab, data_tab, logs_tab = st.tabs(["Assistant", "Patient Data", "Logs & Evaluation"])

with query_tab:
    query = st.text_area(
        "Enter a patient request",
        placeholder="Example: Book a nephrologist for David Thompson and summarize diabetes treatment.",
        height=120,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run Assistant", use_container_width=True)
    with col2:
        clear_button = st.button("Clear History", use_container_width=True)

    if clear_button:
        st.session_state.history = []
        st.rerun()

    if run_button:
        if not query.strip():
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Running agent workflow..."):
                result = assistant.run(query)
            st.session_state.history.insert(0, {"query": query, "result": result})

    if st.session_state.history:
        latest = st.session_state.history[0]["result"]
        st.subheader("Latest Response")
        st.write(latest["response"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tools Called", latest["evaluation"]["tools_called"])
        c2.metric("Successful Tools", latest["evaluation"]["successful_tools"])
        c3.metric("Success Rate", f"{int(latest['evaluation']['success_rate'] * 100)}%")
        c4.metric("Patient Identified", "Yes" if latest["evaluation"]["patient_identified"] else "No")

        with st.expander("Tool Results"):
            st.json(latest["tool_results"])

        with st.expander("Memory Preview"):
            st.write(latest["memory_preview"])

        st.subheader("Recent Queries")
        for item in st.session_state.history[:5]:
            with st.container(border=True):
                st.markdown(f"**Query:** {item['query']}")
                st.write(item["result"]["response"])

with data_tab:
    st.subheader("Patient Registry Preview")
    st.dataframe(assistant.get_patient_preview(), use_container_width=True, hide_index=True)
    st.caption("This table is loaded from records.xlsx and acts as the mock patient registry for the capstone demo.")

with logs_tab:
    if st.session_state.history:
        latest = st.session_state.history[0]["result"]
        st.subheader("Execution Logs")
        for line in latest["logs"]:
            st.code(line)

        st.subheader("Evaluation Summary")
        st.json(latest["evaluation"])
    else:
        st.info("Run the assistant once to view logs and evaluation details.")
