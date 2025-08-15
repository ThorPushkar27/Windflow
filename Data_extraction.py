import os, io, json, uuid, re
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from PIL import Image
from dotenv import load_dotenv


import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv()
genai.configure(api_key="AIzaSyAD20WDiq3fptPQIdT_s5urITyNJ_lGVp4")


class LabResult(BaseModel):
    test_name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    ref_range: Optional[str] = None
    interpretation: Optional[str] = None

class PrescriptionItem(BaseModel):
    medicine_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    notes: Optional[str] = None

class Encounter(BaseModel):
    encounter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["lab_report","prescription"]
    date: Optional[str] = None  # ISO date
    doctor: Optional[str] = None
    diagnosis: Optional[List[str]] = None
    lab_results: Optional[List[LabResult]] = None
    prescriptions: Optional[List[PrescriptionItem]] = None
    source_path: Optional[str] = None

class PatientRecord(BaseModel):
    patient_id: str
    name: Optional[str] = None
    dob: Optional[str] = None
    encounters: List[Encounter]

    @field_validator("encounters")
    def at_least_one_encounter(cls, v):
        assert len(v) > 0
        return v
    
TEST_SYNONYMS = {
    "hb a1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "blood sugar fasting": "Fasting Glucose",
    "fbs": "Fasting Glucose",
    "tsh": "TSH",
    "haemoglobin": "Hemoglobin",
}


DRUG_SYNONYMS = {
    "metformin": "Metformin",
    "atorva": "Atorvastatin",
    "atorvastatin": "Atorvastatin",
    "amlo": "Amlodipine",
    "amlodipine": "Amlodipine",
}


def canon_test(name: str) -> str:
    n = name.strip().lower()
    return TEST_SYNONYMS.get(n, name.strip())

def canon_drug(name: str) -> str:
    n = name.strip().lower()
    return DRUG_SYNONYMS.get(n, name.strip())

MODEL = genai.GenerativeModel('gemini-2.5-pro')

EXTRACTION_SYSTEM_PROMPT = """
You are an expert medical document parser. Return STRICT JSON ONLY.
Schema:
{
  "patient_id": "string (if not present, synthesize stable hash from name+dob or 'unknown')",
  "name": "string|null",
  "dob": "YYYY-MM-DD|null",
  "encounters": [
    {
      "type": "lab_report|prescription",
      "date": "YYYY-MM-DD|null",
      "doctor": "string|null",
      "diagnosis": ["string"]|null,
      "lab_results": [
        {"test_name":"string","value":"string|null","unit":"string|null","ref_range":"string|null","interpretation":"string|null"}
      ]|null,
      "prescriptions": [
        {"medicine_name":"string","dosage":"string|null","frequency":"string|null","duration":"string|null","notes":"string|null"}
      ]|null
    }
  ]
}
Rules:
- If the document is a lab report, populate lab_results; if prescription, populate prescriptions.
- Extract dates if readable; otherwise null.
- Use raw strings; do not invent data. If illegible, set null.
- Return ONLY valid JSON. No prose.
"""

def ocr_and_extract(image_path: str) -> PatientRecord:
    img = Image.open(image_path).convert("RGB")
    # Vision call: image + instructions
    resp = MODEL.generate_content(
        [EXTRACTION_SYSTEM_PROMPT, img],
        safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"},
        generation_config={"temperature": 0}
    )
    raw = resp.text.strip()
    # Some models wrap in ```json ... ```
    raw = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(raw)

    # Ensure encounter type by heuristic (fallback if LLM missed it)
    for enc in data.get("encounters", []):
        if not enc.get("type"):
            if enc.get("lab_results"): enc["type"] = "lab_report"
            elif enc.get("prescriptions"): enc["type"] = "prescription"
            else: enc["type"] = "lab_report"

        # normalize tests & meds
        if enc.get("lab_results"):
            for r in enc["lab_results"]:
                r["test_name"] = canon_test(r["test_name"])
        if enc.get("prescriptions"):
            for p in enc["prescriptions"]:
                p["medicine_name"] = canon_drug(p["medicine_name"])

        enc["source_path"] = image_path

    # Stable patient_id if missing
    if not data.get("patient_id"):
        pid = (data.get("name") or "unknown") + "|" + (data.get("dob") or "unknown")
        data["patient_id"] = f"pid::{uuid.uuid5(uuid.NAMESPACE_DNS, pid)}"

    record = PatientRecord(**data)
    return record


def to_chunks(record: PatientRecord):
    # Build per-encounter text summary
    texts = []
    for enc in record.encounters:
        if enc.type == "lab_report" and enc.lab_results:
            lines = [f"{r.test_name}: {r.value or ''} {r.unit or ''} (ref: {r.ref_range or 'NA'}) - {r.interpretation or ''}".strip()
                     for r in enc.lab_results]
            txt = f"Patient {record.patient_id} lab report on {enc.date or 'unknown'} by {enc.doctor or 'unknown'}:\n" + "\n".join(lines)
            texts.append({"text": txt, "metadata": {"patient_id": record.patient_id, "type": "lab_report", "date": enc.date}})
        if enc.type == "prescription" and enc.prescriptions:
            lines = [f"{p.medicine_name} {p.dosage or ''} {p.frequency or ''} {p.duration or ''}".strip() for p in enc.prescriptions]
            txt = f"Patient {record.patient_id} prescription on {enc.date or 'unknown'} by {enc.doctor or 'unknown'}:\n" + "\n".join(lines)
            texts.append({"text": txt, "metadata": {"patient_id": record.patient_id, "type": "prescription", "date": enc.date}})
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    out = []
    for t in texts:
        for chunk in splitter.split_text(t["text"]):
            out.append({"text": chunk, "metadata": t["metadata"]})
    return out

def process_folder(images_dir: str, events_out="events.jsonl", chunks_out="chunks.jsonl"):
    with open(events_out, "w", encoding="utf-8") as fe, open(chunks_out, "w", encoding="utf-8") as fc:
        for fname in os.listdir(images_dir):
            if fname.lower().endswith((".png",".jpg",".jpeg",".webp",".tif",".tiff")):
                path = os.path.join(images_dir, fname)
                try:
                    rec = ocr_and_extract(path)
                    fe.write(json.dumps(rec.dict()) + "\n")
                    for chunk in to_chunks(rec):
                        fc.write(json.dumps(chunk) + "\n")
                    print("OK:", fname)
                except Exception as e:
                    print("FAIL:", fname, e)


if __name__ == "__main__":
    # Put your downloaded Kaggle images into these folders:
    # - lab reports: ./data/lab_reports
    # - prescriptions: ./data/prescriptions
    process_folder("./data/Lab_report", "events.jsonl", "chunks.jsonl")
    process_folder("./data/Prescription", "events.jsonl", "chunks.jsonl")