import os, json, warnings
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    
)

SCHEMA_CYPHER = """
CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (e:Encounter) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT test_name IF NOT EXISTS FOR (t:Test) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT medicine_name IF NOT EXISTS FOR (m:Medicine) REQUIRE m.name IS UNIQUE;
"""

EVENT_CYPHER = """
WITH $event AS ev
MERGE (p:Patient {id: ev.patient_id})
  ON CREATE SET p.name = ev.name, p.dob = ev.dob
  ON MATCH  SET p.name = coalesce(p.name, ev.name), p.dob = coalesce(p.dob, ev.dob)
WITH p, ev

UNWIND ev.encounters AS enc
MERGE (e:Encounter {id: enc.encounter_id})
  ON CREATE SET e.type = enc.type, e.date = enc.date, e.doctor = enc.doctor, e.source_path = enc.source_path
  ON MATCH  SET e.type = coalesce(e.type, enc.type), e.date = coalesce(e.date, enc.date), e.doctor = coalesce(e.doctor, enc.doctor)
MERGE (p)-[:HAS_ENCOUNTER]->(e)
WITH e, enc

FOREACH (d IN coalesce(enc.diagnosis, []) |
  MERGE (dg:Diagnosis {name: d})
  MERGE (e)-[:HAS_DIAGNOSIS]->(dg)
)
WITH e, enc

FOREACH (lr IN coalesce(enc.lab_results, []) |
  MERGE (t:Test {name: lr.test_name})
  CREATE (r:Result {
      value: lr.value, unit: lr.unit, ref_range: lr.ref_range, interpretation: lr.interpretation
  })
  MERGE (e)-[:HAS_RESULT]->(r)
  MERGE (r)-[:OF_TEST]->(t)
)
WITH e, enc

FOREACH (pr IN coalesce(enc.prescriptions, []) |
  MERGE (m:Medicine {name: pr.medicine_name})
  MERGE (e)-[rel:PRESCRIBED]->(m)
  SET rel.dosage    = coalesce(pr.dosage, rel.dosage),
      rel.frequency = coalesce(pr.frequency, rel.frequency),
      rel.duration  = coalesce(pr.duration, rel.duration),
      rel.notes     = coalesce(pr.notes, rel.notes)
);
"""


def run_schema():
    with driver.session() as s:
        for stmt in SCHEMA_CYPHER.strip().split(";"):
            if stmt.strip():
                s.run(stmt)

def load_events(path="events.jsonl"):
    with driver.session() as s, open(path, "r", encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            s.run(EVENT_CYPHER, event=ev)
    print("Graph load complete.")

if __name__ == "__main__":
    run_schema()
    load_events("events.jsonl")
