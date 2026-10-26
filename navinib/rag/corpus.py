"""Synthetic corpus for the Navinib retrieval layer (built in Checkpoint 3.1).

Everything here is invented. "Product X" is a fictional oncology drug, and the
providers, practices and call notes are synthetic. The label follows the section
layout of a public FDA label so it can be swapped for a real openFDA label later.
"""

LABEL = [  # source = "label", status = "approved"
    ("1 Indications and Usage",
     "Product X is a kinase inhibitor indicated for the treatment of adult patients with "
     "metastatic non-small cell lung cancer (NSCLC) whose tumors have a biomarker Y alteration, "
     "as detected by an FDA-approved test, and who have progressed on or after platinum-based "
     "chemotherapy. Limitation of use: Product X is not indicated for first-line treatment."),
    ("2.1 Recommended Dosage",
     "The recommended dosage of Product X is 160 mg orally once daily, with or without food, "
     "until disease progression or unacceptable toxicity. Swallow tablets whole."),
    ("2.3 Dosage Modifications for Adverse Reactions",
     "For Grade 3 neutropenia (ANC 500 to <1000/mm3), withhold Product X until recovery to "
     "Grade 2 or lower, then resume at 120 mg once daily. For Grade 4 neutropenia, withhold "
     "until recovery and resume at 80 mg once daily. Permanently discontinue for recurrent "
     "Grade 4 neutropenia at 80 mg. For ALT or AST greater than 5 times ULN, withhold until "
     "recovery to Grade 1 and resume at the next lower dose."),
    ("5.1 Neutropenia",
     "Neutropenia occurred in 31% of patients treated with Product X, including Grade 3 or 4 "
     "in 12%. Median time to first onset was 3 weeks. Monitor complete blood counts before "
     "starting Product X, every 2 weeks for the first 2 months, and monthly thereafter. "
     "Patients aged 65 and older had a higher incidence of Grade 3 or 4 neutropenia (17%)."),
    ("5.2 Hepatotoxicity",
     "Elevations of ALT or AST occurred in 22% of patients. Monitor liver function tests "
     "monthly and as clinically indicated. Withhold, reduce the dose or permanently "
     "discontinue based on severity."),
    ("6.1 Clinical Trials Experience",
     "In the pivotal single-arm study (n=212) of patients previously treated with platinum "
     "chemotherapy, the overall response rate was 38% and median duration of response was "
     "9.4 months. The most common adverse reactions (20% or more) were diarrhea, fatigue, "
     "neutropenia, nausea and increased ALT."),
    ("8.5 Geriatric Use",
     "Of the 212 patients in the pivotal study, 41% were 65 years or older. No overall "
     "differences in effectiveness were observed, but Grade 3 or 4 neutropenia was more "
     "frequent in patients aged 65 and older. Follow the monitoring schedule in section 5.1."),
]

TRIALS = [  # source = "trial"; status says whether the use is approved
    ("NCT-SYN-0001", "approved",
     "Long-term follow-up of the pivotal study of Product X in previously treated biomarker Y "
     "NSCLC. Updated median duration of response 10.1 months; no new safety signals. "
     "Neutropenia remained the most common Grade 3 or 4 adverse reaction."),
    ("NCT-SYN-0002", "investigational",
     "Phase 2 study of Product X as first-line therapy in treatment-naive biomarker Y NSCLC. "
     "Preliminary overall response rate 52%. Grade 3 or 4 neutropenia in 15%, higher in "
     "patients over 65. This use is investigational and not approved."),
    ("NCT-SYN-0003", "investigational",
     "Phase 1b study of Product X combined with immunotherapy in small cell lung cancer. "
     "Enrollment ongoing; dose-limiting neutropenia observed at the highest dose level."),
]

# Long-term memory: past call notes. source = "call_note".
CALL_NOTES = [
    ("N-0388", "p-001", "Dr. Alicia Lee", "2026-08-12",
     "Met Dr. Lee at Riverside Oncology Associates. She has three patients on Product X. "
     "Main concern is neutropenia in her older patients; one patient aged 74 had a Grade 3 "
     "episode in week 4. Asked whether there is a simple dose-reduction reference."),
    ("N-0412", "p-001", "Dr. Alicia Lee", "2026-09-23",
     "Follow-up with Dr. Lee. Left the dose-modification card. She said CBC monitoring "
     "every 2 weeks is hard for patients who travel far; asked how other practices manage it. "
     "Nurse navigator Maria Gomez coordinates labs and prefers visits after 2pm."),
    ("N-0371", "p-002", "Dr. Samuel Leigh", "2026-09-30",
     "Dr. Leigh at Northgate Cancer Center asked about first-line data for Product X and "
     "whether the Phase 2 first-line results will be published. Told him the first-line use "
     "is not approved and referred the question to Medical Information."),
    ("N-0405", "p-003", "Dr. Helen Smith", "2026-10-02",
     "Dr. Smith is interested in liver function monitoring; one patient had ALT above 5x ULN. "
     "Reviewed section 5.2. Prefers short visits, 15 minutes maximum."),
    ("N-0420", "p-004", "Dr. Daniel Okafor", "2026-10-07",
     "Dr. Okafor has not prescribed Product X yet. Wants response-rate and duration data in "
     "previously treated patients. Busy clinic, best reached before 10am."),
]
