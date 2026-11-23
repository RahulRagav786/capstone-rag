"""Generate SYNTHETIC blood-report PDFs (CBC, HbA1c, lipids, kidney, thyroid) plus a
reference guide PDF and a retrieval evaluation set.

Every value is randomly generated. No real patient data is used.

Outputs:
  data/reports/R0001_P0001.pdf ...            one multi-page report per patient
  data/reports/manifest.jsonl                  ground truth: which test is on which page
  data/reference/lab_test_guide.pdf            general explainer text (not patient-specific)
  data/eval/retrieval_questions.jsonl          questions with gold (patient, test, page)

Usage:
  python scripts/make_synthetic_reports.py --n 200 --seed 42
"""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Test catalogue: (test name, unit, low, high) — sex-specific where it matters.
# Standard adult reference intervals; exact cut-offs vary by lab.
# ---------------------------------------------------------------------------
PANELS = {
    "Complete Blood Count (CBC)": [
        ("Hemoglobin", "g/dL", {"M": (13.0, 17.0), "F": (12.0, 15.5)}),
        ("RBC Count", "x10^6/uL", {"M": (4.5, 5.9), "F": (4.0, 5.2)}),
        ("Hematocrit (PCV)", "%", {"M": (40.0, 50.0), "F": (36.0, 46.0)}),
        ("MCV", "fL", (80.0, 100.0)),
        ("MCH", "pg", (27.0, 33.0)),
        ("MCHC", "g/dL", (32.0, 36.0)),
        ("Total WBC Count", "x10^3/uL", (4.0, 11.0)),
        ("Neutrophils", "%", (40.0, 75.0)),
        ("Lymphocytes", "%", (20.0, 45.0)),
        ("Platelet Count", "x10^3/uL", (150.0, 410.0)),
    ],
    "Diabetes Panel": [
        ("HbA1c (Glycated Hemoglobin)", "%", (4.0, 5.6)),
        ("Estimated Average Glucose (eAG)", "mg/dL", (70.0, 116.0)),
        ("Fasting Plasma Glucose", "mg/dL", (70.0, 99.0)),
    ],
    "Lipid Profile": [
        ("Total Cholesterol", "mg/dL", (None, 200.0)),
        ("LDL Cholesterol", "mg/dL", (None, 100.0)),
        ("HDL Cholesterol", "mg/dL", {"M": (40.0, None), "F": (50.0, None)}),
        ("Triglycerides", "mg/dL", (None, 150.0)),
    ],
    "Kidney Function Test": [
        ("Serum Creatinine", "mg/dL", {"M": (0.7, 1.3), "F": (0.6, 1.1)}),
        ("Blood Urea Nitrogen (BUN)", "mg/dL", (7.0, 20.0)),
    ],
    "Thyroid Profile": [
        ("TSH", "mIU/L", (0.4, 4.0)),
    ],
}

# Patient "profiles" shift some values so the corpus contains realistic abnormal patterns.
PROFILES = ["healthy", "healthy", "prediabetic", "diabetic", "anemic", "dyslipidemic", "hypothyroid", "mixed"]

FIRST = ["Aarav", "Diya", "Rohan", "Ananya", "Vikram", "Isha", "Karan", "Meera", "Arjun", "Sneha",
         "Rahul", "Priya", "Aditya", "Kavya", "Siddharth", "Neha", "Manish", "Pooja", "Nikhil", "Riya"]
LAST = ["Sharma", "Iyer", "Patel", "Reddy", "Gupta", "Nair", "Singh", "Das", "Menon", "Joshi"]

# Natural-language ways a patient might ask about each test (used for the eval set).
QUESTION_ALIASES = {
    "Hemoglobin": ["hemoglobin", "Hb level", "haemoglobin"],
    "RBC Count": ["red blood cell count", "RBC count"],
    "Hematocrit (PCV)": ["hematocrit", "PCV"],
    "MCV": ["MCV", "mean corpuscular volume"],
    "MCH": ["MCH"],
    "MCHC": ["MCHC"],
    "Total WBC Count": ["white blood cell count", "WBC"],
    "Neutrophils": ["neutrophils"],
    "Lymphocytes": ["lymphocytes"],
    "Platelet Count": ["platelets", "platelet count"],
    "HbA1c (Glycated Hemoglobin)": ["HbA1c", "A1c", "3-month average sugar", "glycated hemoglobin"],
    "Estimated Average Glucose (eAG)": ["estimated average glucose", "eAG"],
    "Fasting Plasma Glucose": ["fasting sugar", "fasting blood glucose", "fasting glucose"],
    "Total Cholesterol": ["total cholesterol", "cholesterol"],
    "LDL Cholesterol": ["LDL", "bad cholesterol"],
    "HDL Cholesterol": ["HDL", "good cholesterol"],
    "Triglycerides": ["triglycerides"],
    "Serum Creatinine": ["creatinine", "kidney creatinine"],
    "Blood Urea Nitrogen (BUN)": ["BUN", "urea"],
    "TSH": ["TSH", "thyroid"],
}
QUESTION_TEMPLATES = [
    "What is my {alias}?",
    "Is my {alias} normal?",
    "Explain my {alias} result.",
    "Should I be worried about my {alias}?",
    "What does my report say about {alias}?",
]


def ref_range(spec, sex):
    return spec[sex] if isinstance(spec, dict) else spec


def fmt_range(lo, hi):
    if lo is None:
        return f"< {hi:g}"
    if hi is None:
        return f"> {lo:g}"
    return f"{lo:g} - {hi:g}"


def sample_value(test, lo, hi, profile, rng, state):
    """Draw a value: mostly in range, pushed out of range according to the profile."""
    def normal():
        a = lo if lo is not None else hi * 0.6
        b = hi if hi is not None else lo * 1.6
        return rng.uniform(a + (b - a) * 0.1, b - (b - a) * 0.1)

    v = normal()
    if profile in ("anemic", "mixed") and test in ("Hemoglobin", "Hematocrit (PCV)", "RBC Count", "MCV", "MCH"):
        v = lo * rng.uniform(0.70, 0.95)
    if test == "HbA1c (Glycated Hemoglobin)":
        v = {"prediabetic": rng.uniform(5.7, 6.4), "diabetic": rng.uniform(6.5, 10.5),
             "mixed": rng.uniform(6.0, 8.0)}.get(profile, rng.uniform(4.6, 5.5))
        state["a1c"] = v
    if test == "Estimated Average Glucose (eAG)":
        v = 28.7 * state["a1c"] - 46.7          # standard ADAG formula
    if test == "Fasting Plasma Glucose":
        v = {"prediabetic": rng.uniform(100, 125), "diabetic": rng.uniform(126, 220),
             "mixed": rng.uniform(105, 160)}.get(profile, rng.uniform(75, 98))
    if profile in ("dyslipidemic", "mixed"):
        if test == "Total Cholesterol":
            v = rng.uniform(205, 290)
        if test == "LDL Cholesterol":
            v = rng.uniform(110, 190)
        if test == "Triglycerides":
            v = rng.uniform(155, 320)
        if test == "HDL Cholesterol":
            v = lo * rng.uniform(0.7, 0.98)
    if profile == "hypothyroid" and test == "TSH":
        v = rng.uniform(4.5, 12.0)
    # occasional random outlier anywhere, so "healthy" patients are not perfectly clean
    if rng.random() < 0.04:
        v = v * rng.choice([0.8, 1.25])
    return round(v, 1 if v >= 10 else 2)


def flag(v, lo, hi):
    if lo is not None and v < lo:
        return "LOW"
    if hi is not None and v > hi:
        return "HIGH"
    return ""


def build_report(idx, rng, out_dir, styles):
    pid, rid = f"P{idx:04d}", f"R{idx:04d}"
    sex = rng.choice(["M", "F"])
    age = rng.randint(22, 78)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    profile = rng.choice(PROFILES)
    rdate = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))

    # CBC is always page 1; the other panels are shuffled across pages 2-3 so page numbers vary.
    others = [p for p in PANELS if p != "Complete Blood Count (CBC)"]
    rng.shuffle(others)
    pages = [["Complete Blood Count (CBC)"], others[:2], others[2:]]

    state, story, tests_meta = {}, [], []
    header = (f"<b>SYNTHETIC DIAGNOSTICS LAB</b> &nbsp; (synthetic data - not a real patient)<br/>"
              f"Patient: {name} &nbsp; | &nbsp; Patient ID: {pid} &nbsp; | &nbsp; Age/Sex: {age}/{sex}<br/>"
              f"Report ID: {rid} &nbsp; | &nbsp; Report Date: {rdate.isoformat()}")
    abnormal = []
    for page_no, panel_names in enumerate(pages, start=1):
        story.append(Paragraph(header, styles["Normal"]))
        story.append(Spacer(1, 12))
        for panel in panel_names:
            story.append(Paragraph(f"<b>{panel}</b>", styles["Heading3"]))
            rows = [["Test", "Result", "Unit", "Reference Range", "Flag"]]
            for test, unit, spec in PANELS[panel]:
                lo, hi = ref_range(spec, sex)
                v = sample_value(test, lo, hi, profile, rng, state)
                f = flag(v, lo, hi)
                if f:
                    abnormal.append(f"{test} {f.lower()}")
                rows.append([test, f"{v:g}", unit, fmt_range(lo, hi), f])
                tests_meta.append({"test": test, "panel": panel, "page": page_no,
                                   "value": v, "unit": unit, "flag": f})
            t = Table(rows, colWidths=[170, 60, 70, 110, 50])
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))
        if page_no == len(pages):
            remark = ("Pathologist remarks: " + (
                "Values outside the reference range: " + ", ".join(abnormal) +
                ". Please correlate clinically and consult your physician."
                if abnormal else "All tested parameters are within reference ranges."))
            story.append(Paragraph(remark, styles["Normal"]))
        else:
            story.append(PageBreak())

    path = out_dir / f"{rid}_{pid}.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, title=f"Lab report {rid}").build(story)
    return {"report_id": rid, "patient_id": pid, "sex": sex, "age": age, "profile": profile,
            "report_date": rdate.isoformat(), "file": path.name, "tests": tests_meta}


# General, non-diagnostic explanations used as a shared "reference" corpus.
REFERENCE_GUIDE = {
    "Hemoglobin": "Hemoglobin is the protein in red blood cells that carries oxygen. Low hemoglobin (anemia) can cause tiredness and breathlessness; common causes include iron, vitamin B12 or folate deficiency and blood loss. High values can occur with dehydration, smoking or living at altitude.",
    "RBC Count, Hematocrit and Red Cell Indices (MCV, MCH, MCHC)": "RBC count is the number of red blood cells. Hematocrit (PCV) is the percentage of blood volume made of red cells. MCV is the average size of red cells: low MCV (small cells) is typical of iron-deficiency anemia, high MCV (large cells) of B12 or folate deficiency. MCH and MCHC describe how much hemoglobin each cell carries.",
    "White Blood Cells, Neutrophils and Lymphocytes": "White blood cells fight infection. A high WBC count often reflects infection or inflammation; a low count can follow viral illness or some medicines. Neutrophils mainly fight bacterial infections; lymphocytes are involved in viral infections and immunity.",
    "Platelet Count": "Platelets help blood clot. Low platelets can increase bruising and bleeding; high platelets can occur with inflammation, infection or iron deficiency.",
    "HbA1c (Glycated Hemoglobin)": "HbA1c reflects average blood sugar over roughly the past 2 to 3 months, because glucose attaches to hemoglobin in red cells that live about 120 days. Commonly used cut-offs: below 5.7% is normal, 5.7% to 6.4% indicates prediabetes, and 6.5% or above is consistent with diabetes (diagnosis is usually confirmed with a repeat test). Estimated Average Glucose (eAG) converts HbA1c into mg/dL using eAG = 28.7 x HbA1c - 46.7.",
    "Fasting Plasma Glucose": "Fasting plasma glucose is blood sugar after at least 8 hours without food. 70 to 99 mg/dL is normal, 100 to 125 mg/dL suggests impaired fasting glucose (prediabetes), and 126 mg/dL or higher on repeat testing is consistent with diabetes.",
    "Lipid Profile": "Total cholesterol below 200 mg/dL is desirable. LDL ('bad') cholesterol builds up in artery walls; lower is better, and targets depend on overall heart risk. HDL ('good') cholesterol helps remove cholesterol; higher is better (above 40 mg/dL for men and above 50 mg/dL for women). Triglycerides below 150 mg/dL are normal and rise with high sugar intake, alcohol and uncontrolled diabetes.",
    "Kidney Function (Creatinine and BUN)": "Serum creatinine is a waste product filtered by the kidneys; raised creatinine can indicate reduced kidney function, though muscle mass and hydration also affect it. Blood urea nitrogen (BUN) rises with dehydration, high protein intake or reduced kidney function.",
    "TSH (Thyroid Stimulating Hormone)": "TSH is released by the pituitary gland to control the thyroid. A high TSH usually means the thyroid is underactive (hypothyroidism); a low TSH can mean it is overactive (hyperthyroidism).",
    "Important note": "Reference ranges vary between laboratories. A single out-of-range value is not a diagnosis. Results should always be interpreted by a qualified doctor together with symptoms and history.",
}


def build_reference(out_dir, styles):
    story = [Paragraph("<b>Guide to Common Blood Tests</b> (general information, not medical advice)", styles["Title"])]
    for title, body in REFERENCE_GUIDE.items():
        story += [Paragraph(f"<b>{title}</b>", styles["Heading3"]), Paragraph(body, styles["Normal"]), Spacer(1, 8)]
    path = out_dir / "lab_test_guide.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, title="Guide to Common Blood Tests").build(story)


def build_eval(manifest, rng, per_report=3):
    rows = []
    for rep in manifest:
        # bias towards abnormal tests: those are what patients ask about
        abnormal = [t for t in rep["tests"] if t["flag"]]
        normal = [t for t in rep["tests"] if not t["flag"]]
        picks = rng.sample(abnormal, min(2, len(abnormal)))
        picks += rng.sample(normal, per_report - len(picks))
        for t in picks:
            alias = rng.choice(QUESTION_ALIASES[t["test"]])
            rows.append({"patient_id": rep["patient_id"], "report_id": rep["report_id"],
                         "question": rng.choice(QUESTION_TEMPLATES).format(alias=alias),
                         "gold_test": t["test"], "gold_page": t["page"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of patient reports")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    styles = getSampleStyleSheet()
    rep_dir, ref_dir, eval_dir = ROOT / "data/reports", ROOT / "data/reference", ROOT / "data/eval"
    for d in (rep_dir, ref_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest = [build_report(i, rng, rep_dir, styles) for i in range(1, args.n + 1)]
    with open(rep_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    build_reference(ref_dir, styles)
    qs = build_eval(manifest, rng)
    with open(eval_dir / "retrieval_questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(manifest)} reports, 1 reference guide, {len(qs)} eval questions.")


if __name__ == "__main__":
    main()
