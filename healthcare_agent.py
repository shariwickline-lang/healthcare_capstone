from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import openpyxl
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_FILES = {
    "Rebeca Nagle": DATA_DIR / "sample_patient.pdf",
    "Anjali Mehra": DATA_DIR / "sample_report_anjali.pdf",
    "David Thompson": DATA_DIR / "sample_report_david.pdf",
    "Ramesh Kulkarni": DATA_DIR / "sample_report_ramesh.pdf",
}

DEMO_MEDICAL_KB = {
    "chronic kidney disease": (
        "Chronic kidney disease (CKD) is managed by slowing progression, controlling blood pressure, "
        "reviewing diabetes status, monitoring kidney function, and coordinating nephrology follow-up. "
        "Common care elements include ACE inhibitor or ARB therapy when appropriate, dietary guidance, "
        "lab monitoring, medication review, and escalation for swelling, confusion, chest pain, or worsening urine output."
    ),
    "type 2 diabetes": (
        "Type 2 diabetes care commonly includes glucose-lowering medication, lifestyle counseling, HbA1c monitoring, "
        "kidney function review, nutrition support, and foot/eye screening. Concerning symptoms include severe thirst, "
        "vomiting, confusion, chest pain, or signs of hypoglycemia."
    ),
    "hypertension": (
        "Hypertension management focuses on blood pressure control, medication adherence, sodium reduction, exercise, "
        "routine labs, and follow-up visits. Urgent review is needed for severe headache, chest pain, neurological symptoms, "
        "or markedly elevated blood pressure."
    ),
    "upper respiratory infection": (
        "Most uncomplicated upper respiratory infections are managed with rest, fluids, symptom relief, and monitoring. "
        "Clinical reassessment is warranted for shortness of breath, persistent fever, dehydration, or worsening symptoms."
    ),
}

SPECIALTY_KEYWORDS = {
    "nephrologist": "nephrology",
    "kidney": "nephrology",
    "renal": "nephrology",
    "diabetologist": "endocrinology",
    "diabetes": "endocrinology",
    "cardiologist": "cardiology",
    "heart": "cardiology",
    "pulmonologist": "pulmonology",
    "lung": "pulmonology",
    "family": "family medicine",
}

DOCTOR_DB = {
    "general": [{"name": "Dr. Priya Sharma", "hospital": "City Health Clinic"}],
    "nephrology": [{"name": "Dr. Anand Mehta", "hospital": "Apollo Hospital"}],
    "endocrinology": [{"name": "Dr. Sunita Rao", "hospital": "Fortis Hospital"}],
    "cardiology": [{"name": "Dr. Vikram Nair", "hospital": "Max Hospital"}],
    "pulmonology": [{"name": "Dr. Kavita Joshi", "hospital": "Columbia Asia"}],
    "family medicine": [{"name": "Dr. Megana Lanoi", "hospital": "Bridport Family Medicine"}],
}


class AgentState(TypedDict):
    query: str
    patient_name: str
    tool_results: List[Dict[str, Any]]
    final_response: str
    tools_to_run: List[str]
    logs: List[str]


@dataclass
class PatientMemory:
    embedding_model: HuggingFaceEmbeddings
    stores: Dict[str, FAISS] = field(default_factory=dict)

    def save(self, patient_name: str, query: str, response: str) -> None:
        text = f"[Session] Q: {query}\nA: {response[:700]}"
        metadata = [{"patient": patient_name, "ts": datetime.now().isoformat()}]
        if patient_name not in self.stores:
            self.stores[patient_name] = FAISS.from_texts([text], self.embedding_model, metadatas=metadata)
        else:
            self.stores[patient_name].add_texts([text], metadatas=metadata)

    def retrieve(self, patient_name: str, query: str, k: int = 2) -> str:
        if patient_name not in self.stores:
            return "No prior session history for this patient."
        docs = self.stores[patient_name].similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)


class HealthcareAssistant:
    """Simple agentic healthcare demo for a Streamlit capstone app."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.parser = StrOutputParser()
        self._validate_files()

        self.patient_df = self._load_patient_dataframe()
        self.patient_registry = self._build_patient_registry(self.patient_df)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.patient_vectorstore = self._build_patient_vectorstore()
        self.patient_retriever = self.patient_vectorstore.as_retriever(search_kwargs={"k": 4})
        self.memory = PatientMemory(self.embeddings)
        self.llm = self._build_llm()
        self.agent = self._compile_graph()

    def _validate_files(self) -> None:
        required = [self.data_dir / "records.xlsx", *PDF_FILES.values()]
        missing = [str(path) for path in required if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required project files: {missing}")

    def _build_llm(self) -> ChatOpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it in Streamlit secrets or your local environment."
            )
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def _load_patient_dataframe(self) -> pd.DataFrame:
        workbook = openpyxl.load_workbook(self.data_dir / "records.xlsx")
        sheet = workbook.active
        headers = [cell.value for cell in sheet[1]]
        rows = [dict(zip(headers, row)) for row in sheet.iter_rows(min_row=2, values_only=True)]
        return pd.DataFrame(rows)

    def _build_patient_registry(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        registry: Dict[str, Dict[str, Any]] = {}
        for row in df.to_dict(orient="records"):
            name = row.get("Name")
            phone = row.get("Phone_number")
            if name:
                registry[str(name).lower().strip()] = row
            if phone:
                registry[str(phone).strip()] = row
        return registry

    def _extract_pdf_text(self, path: Path) -> str:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _build_patient_vectorstore(self) -> FAISS:
        texts: List[str] = []
        metadata: List[Dict[str, str]] = []
        for patient_name, path in PDF_FILES.items():
            text = self._extract_pdf_text(path)
            chunks = self._chunk_text(text, chunk_size=500, chunk_overlap=80)
            texts.extend(chunks)
            metadata.extend([{"patient": patient_name}] * len(chunks))
        return FAISS.from_texts(texts, self.embeddings, metadatas=metadata)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
        return chunks

    def lookup_patient(self, name: str) -> Dict[str, Any]:
        key = name.lower().strip()
        if key in self.patient_registry:
            patient = self.patient_registry[key]
            return {
                "status": "found",
                "name": patient.get("Name"),
                "age": patient.get("Age"),
                "gender": patient.get("Gender"),
                "phone": str(patient.get("Phone_number", "")),
                "address": patient.get("Address"),
                "summary": patient.get("Summary") or "No summary stored in registry.",
            }
        for registry_key, patient in self.patient_registry.items():
            if key in registry_key or registry_key in key:
                return {
                    "status": "found",
                    "name": patient.get("Name"),
                    "age": patient.get("Age"),
                    "gender": patient.get("Gender"),
                    "phone": str(patient.get("Phone_number", "")),
                    "address": patient.get("Address"),
                    "summary": patient.get("Summary") or "No summary stored in registry.",
                }
        return {"status": "not_found", "message": f"No patient named {name} in registry."}

    def retrieve_medical_history(self, patient_name: str) -> Dict[str, Any]:
        query = f"medical history diagnosis medications treatment plan for {patient_name}"
        docs = self.patient_retriever.invoke(query)
        relevant = [doc for doc in docs if patient_name.split()[0].lower() in doc.page_content.lower()]
        context = "\n\n".join(doc.page_content for doc in (relevant or docs))
        prompt = ChatPromptTemplate.from_template(
            "You are a medical summarization assistant for an educational capstone demo.\n"
            "Summarize the clinical notes for patient: {patient_name}.\n"
            "Use only the provided context.\n"
            "Include: Diagnoses, Medications, Lab Results (if any), Vitals, Treatment Plan.\n\n"
            "Clinical Notes:\n{context}\n\n"
            "Return a short, structured summary."
        )
        summary = (prompt | self.llm | self.parser).invoke({"patient_name": patient_name, "context": context})
        return {"status": "found", "patient": patient_name, "summary": summary}

    def _infer_specialty(self, query: str) -> str:
        lowered = query.lower()
        for keyword, specialty in SPECIALTY_KEYWORDS.items():
            if keyword in lowered:
                return specialty
        return "general"

    def book_appointment(self, patient_name: str, specialty: str, preference: str = "morning") -> Dict[str, Any]:
        doctor = random.choice(DOCTOR_DB.get(specialty.lower(), DOCTOR_DB["general"]))
        visit_date = (datetime.now() + timedelta(days=random.randint(2, 5))).strftime("%A, %d %B %Y")
        visit_time = "10:30 AM" if "morning" in preference.lower() else "3:00 PM"
        ref = f"APT-{random.randint(10000, 99999)}"
        return {
            "status": "confirmed",
            "booking_ref": ref,
            "patient": patient_name,
            "doctor": doctor["name"],
            "hospital": doctor["hospital"],
            "specialty": specialty,
            "date": visit_date,
            "time": visit_time,
        }

    def _extract_topic(self, query: str) -> str:
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract the main medical condition or topic from the user query below.\n"
            "If a clear topic is not present, reply with 'general health follow-up'.\n\n"
            "Query: {query}"
        )
        topic = (extract_prompt | self.llm | self.parser).invoke({"query": query}).strip()
        return topic or "general health follow-up"

    def search_medical_info(self, topic: str) -> Dict[str, Any]:
        topic_lower = topic.lower()
        matched_context = None
        for key, value in DEMO_MEDICAL_KB.items():
            if key in topic_lower or topic_lower in key:
                matched_context = value
                topic = key.title()
                break
        if matched_context is None:
            matched_context = (
                "General educational guidance: review symptoms, current medications, relevant history, red flags, "
                "and the need for clinician follow-up. This demo does not replace professional medical advice."
            )
        prompt = ChatPromptTemplate.from_template(
            "You are a clinical information assistant for an educational capstone demo.\n"
            "Using only the supplied context, summarize the topic below.\n"
            "Include: Overview, Common Care Approaches, Monitoring Points, Red Flags.\n"
            "End with: 'This demo summary is not a diagnosis.'\n\n"
            "Topic: {topic}\n\n"
            "Context:\n{context}"
        )
        result = (prompt | self.llm | self.parser).invoke({"topic": topic, "context": matched_context})
        return {"status": "found", "topic": topic, "information": result}

    def update_patient_summary(self, patient_name: str, new_summary: str) -> Dict[str, Any]:
        key = patient_name.lower().strip()
        if key in self.patient_registry:
            self.patient_registry[key]["Summary"] = new_summary
            return {"status": "updated", "patient": patient_name}
        return {"status": "not_found", "message": f"Patient {patient_name} not found."}

    def _heuristic_route(self, query: str) -> Dict[str, Any]:
        lowered = query.lower()
        tools: List[str] = []
        patient_name = "unknown"

        for candidate in self.patient_df["Name"].dropna().astype(str).tolist():
            if candidate.lower() in lowered:
                patient_name = candidate
                tools.append("lookup_patient")
                break

        if any(word in lowered for word in ["history", "diagnosis", "treatment plan", "medication", "summary"]):
            tools.append("retrieve_history")
        if any(word in lowered for word in ["book", "schedule", "appointment"]):
            tools.append("book_appointment")
        if any(word in lowered for word in ["treatment", "guideline", "latest", "information", "explain"]):
            tools.append("search_medical_info")
        if any(word in lowered for word in ["update", "change", "add to record"]):
            tools.append("update_record")

        if not tools:
            tools = ["lookup_patient"]
        return {"patient_name": patient_name, "tools": list(dict.fromkeys(tools))}

    def route_query(self, state: AgentState) -> AgentState:
        router_prompt = ChatPromptTemplate.from_template(
            "You are a healthcare assistant router for an educational capstone demo.\n"
            "Analyze the query and decide which tools to call.\n\n"
            "User Query: {query}\n\n"
            "Available tools:\n"
            "- lookup_patient: when the query mentions a patient name\n"
            "- retrieve_history: when the query asks for medical history, diagnosis, medications, or treatment plan\n"
            "- book_appointment: when the query asks to book or schedule a doctor\n"
            "- search_medical_info: when the query asks for disease information or treatment guidance\n"
            "- update_record: when the query asks to update patient information\n\n"
            "Return only valid JSON in this exact shape:\n"
            '{"patient_name": "<name or unknown>", "tools": ["tool1", "tool2"]}'
        )
        logs = list(state.get("logs", []))
        try:
            raw = (router_prompt | self.llm | self.parser).invoke({"query": state["query"]}).strip()
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
        except Exception:
            parsed = self._heuristic_route(state["query"])
            logs.append("Router fallback used due to JSON parsing issue.")
        patient_name = parsed.get("patient_name", "unknown") or "unknown"
        tools = parsed.get("tools", ["lookup_patient"])
        logs.append(f"Router selected patient='{patient_name}' and tools={tools}")
        return {
            **state,
            "patient_name": patient_name,
            "tools_to_run": tools,
            "tool_results": [],
            "logs": logs,
        }

    def execute_tools(self, state: AgentState) -> AgentState:
        results: List[Dict[str, Any]] = []
        logs = list(state.get("logs", []))
        patient_name = state.get("patient_name", "unknown")
        query = state.get("query", "")

        for tool_name in state.get("tools_to_run", []):
            try:
                if tool_name == "lookup_patient":
                    result = self.lookup_patient(patient_name)
                elif tool_name == "retrieve_history":
                    result = self.retrieve_medical_history(patient_name)
                elif tool_name == "book_appointment":
                    specialty = self._infer_specialty(query)
                    result = self.book_appointment(patient_name, specialty)
                elif tool_name == "search_medical_info":
                    topic = self._extract_topic(query)
                    result = self.search_medical_info(topic)
                elif tool_name == "update_record":
                    result = {"status": "skipped", "note": "Requires explicit new data in this simple demo."}
                else:
                    result = {"status": "unknown_tool"}
                logs.append(f"Tool '{tool_name}' completed with status='{result.get('status', 'done')}'")
            except Exception as exc:
                result = {"status": "error", "message": str(exc)}
                logs.append(f"Tool '{tool_name}' failed: {exc}")
            results.append({"tool": tool_name, "result": result})

        return {**state, "tool_results": results, "logs": logs}

    def aggregate_response(self, state: AgentState) -> AgentState:
        tool_json = json.dumps(state["tool_results"], indent=2)
        memory_context = ""
        patient_name = state.get("patient_name", "unknown")
        if patient_name != "unknown":
            memory_context = self.memory.retrieve(patient_name, state["query"])

        prompt = ChatPromptTemplate.from_template(
            "You are a healthcare assistant for an educational capstone demo.\n"
            "Compose a clear and professional response to the user.\n"
            "Do not mention tools or internal system steps.\n"
            "If an appointment is booked, show the details clearly.\n"
            "If history is available, summarize it in readable language.\n"
            "If medical information is available, use short bullet points.\n"
            "If prior memory is available, briefly use it for continuity.\n"
            "End with a short safety note that this is a demo assistant and not a substitute for medical care.\n\n"
            "Original Query:\n{query}\n\n"
            "Prior Memory:\n{memory_context}\n\n"
            "Tool Results:\n{tool_results}"
        )
        response = (prompt | self.llm | self.parser).invoke(
            {
                "query": state["query"],
                "memory_context": memory_context,
                "tool_results": tool_json,
            }
        )
        return {**state, "final_response": response}

    def _compile_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route_query", self.route_query)
        graph.add_node("execute_tools", self.execute_tools)
        graph.add_node("aggregate_response", self.aggregate_response)
        graph.set_entry_point("route_query")
        graph.add_edge("route_query", "execute_tools")
        graph.add_edge("execute_tools", "aggregate_response")
        graph.add_edge("aggregate_response", END)
        return graph.compile()

    def run(self, query: str) -> Dict[str, Any]:
        initial_state: AgentState = {
            "query": query,
            "patient_name": "",
            "tool_results": [],
            "final_response": "",
            "tools_to_run": [],
            "logs": [],
        }
        final_state = self.agent.invoke(initial_state)
        patient_name = final_state.get("patient_name", "unknown")
        if patient_name and patient_name != "unknown":
            self.memory.save(patient_name, query, final_state["final_response"])

        success_count = sum(
            1 for item in final_state["tool_results"] if item["result"].get("status") in {"found", "confirmed", "updated"}
        )
        total_tools = max(1, len(final_state["tool_results"]))
        evaluation = {
            "tools_called": len(final_state["tool_results"]),
            "successful_tools": success_count,
            "success_rate": round(success_count / total_tools, 2),
            "patient_identified": patient_name != "unknown",
        }
        return {
            "response": final_state["final_response"],
            "patient_name": patient_name,
            "tool_results": final_state["tool_results"],
            "logs": final_state["logs"],
            "evaluation": evaluation,
            "memory_preview": self.memory.retrieve(patient_name, query) if patient_name != "unknown" else "No memory available.",
        }

    def get_patient_preview(self) -> pd.DataFrame:
        return self.patient_df[["Name", "Age", "Gender", "Address", "Summary"]].copy()
