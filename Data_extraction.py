
import os, json, uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from dotenv import load_dotenv

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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
    date: Optional[str] = None
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


ocr_model = ocr_predictor(pretrained=True)

def run_ocr(image_path: str) -> str:
    doc = DocumentFile.from_images(image_path)
    result = ocr_model(doc)
    return result.render()  # raw text



def gemini_flash_generate(prompt: str) -> str:
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 2048,
        }
    )
    return response.text

class GeminiLLM:
    def __call__(self, prompt: str) -> str:
        return gemini_flash_generate(prompt)

llm = GeminiLLM()
parser = PydanticOutputParser(pydantic_object=PatientRecord)

prompt = PromptTemplate(
    template=(
        "You are an expert medical document parser.\n"
        "Extract structured information from this OCR text.\n\n"
        "{format_instructions}\n\n"
        "OCR TEXT:\n{doc}\n"
    ),
    input_variables=["doc"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def ocr_and_extract(image_path: str) -> PatientRecord:
    
    text = run_ocr(image_path)

    
    _prompt = prompt.format(doc=text)
    raw_output = llm(_prompt)

    
    record = parser.parse(raw_output)

    
    for enc in record.encounters:
        if not enc.type:
            if enc.lab_results: enc.type = "lab_report"
            elif enc.prescriptions: enc.type = "prescription"
            else: enc.type = "lab_report"

        if enc.lab_results:
            for r in enc.lab_results:
                r.test_name = canon_test(r.test_name)

        if enc.prescriptions:
            for p in enc.prescriptions:
                p.medicine_name = canon_drug(p.medicine_name)

        enc.source_path = image_path

    if not record.patient_id:
        pid = (record.name or "unknown") + "|" + (record.dob or "unknown")
        record.patient_id = f"pid::{uuid.uuid5(uuid.NAMESPACE_DNS, pid)}"

    return record


def to_chunks(record: PatientRecord):
    texts = []
    for enc in record.encounters:
        if enc.type == "lab_report" and enc.lab_results:
            lines = [f"{r.test_name}: {r.value or ''} {r.unit or ''} (ref: {r.ref_range or 'NA'}) - {r.interpretation or ''}".strip()
                     for r in enc.lab_results]
            txt = f"Patient {record.patient_id} lab report on {enc.date or 'unknown'} by {enc.doctor or 'unknown'}:\n" + "\n".join(lines)
            texts.append({"text": txt, "metadata": {"patient_id": record.patient_id, "type": "lab_report", "date": enc.date}})
        if enc.type == "prescription" and enc.prescriptions:
            lines = [f"{p.medicine_name} {p.dosage or ''} {p.frequency or ''} {p.duration or ''}".strip()
                     for p in enc.prescriptions]
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
                    fe.write(json.dumps(rec.model_dump()) + "\n")
                    for chunk in to_chunks(rec):
                        fc.write(json.dumps(chunk) + "\n")
                    print(" OK:", fname)
                except Exception as e:
                    print(" FAIL:", fname, e)


if __name__ == "__main__":
    process_folder("./data/Lab_report", "events.jsonl", "chunks.jsonl")
    process_folder("./data/Prescription", "events.jsonl", "chunks.jsonl")
    
    

