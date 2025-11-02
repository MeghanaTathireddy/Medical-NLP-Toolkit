# Physician Notetaker \

This folder contains the code and resources for all three tasks:  
**Medical NLP Summarization, Sentiment & Intent Analysis, and SOAP Note Generation.**

## Folder Structure

```
assignment_questions_answers.txt
README.md
requirements.txt
task1_medical_ner.py
task2_sentiment_intent.py
task3_soap_generation.py
```

## Setup Instructions

1. **Unzip** this folder to your local machine.
2. **Install dependencies** (Python 3.8+ recommended):

    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    # (Optional for better medical NER)
    # pip install scispacy
    # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
    ```

## How to Run Each Task

### Task 1 — Medical NLP Summarization

You can import and use the [`process_task1`](task1_medical_ner.py) function directly:

```python
from task1_medical_ner import process_task1
import json

conversation = """
Physician: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""

print(json.dumps(process_task1(conversation), indent=2))
```

Or run the file directly for a demo:

```bash
python task1_medical_ner.py
```

---

### Task 2 — Sentiment & Intent Analysis

Run from the command line with a patient statement:

```bash
python task2_sentiment_intent.py --text "I'm worried about my back pain"
```

Or import [`process_task2`](task2_sentiment_intent.py):

```python
from task2_sentiment_intent import process_task2
print(process_task2("I'm worried about my back pain"))
```

---

### Task 3 — SOAP Note Generation

Run the demo script:

```bash
python task3_soap_generation.py
```

Or import [`process_task3`](task3_soap_generation.py):

```python
from task3_soap_generation import process_task3
result = process_task3("...conversation text...")
print(result["formatted_note"])
```

---

## Notes for Evaluators

- All imports like `from task1_medical_ner import process_task1` work because the `.py` files are in the same directory.
- No extra modules or folders are required; everything is self-contained.
- Sample conversations and expected outputs are included in each script's `__main__` block.
- See [assignment_questions_answers.txt](assignment_questions_answers.txt) for detailed answers to assignment questions.

---

## Quick Smoke Tests

```bash
python task2_sentiment_intent.py --text "I'm worried about my back pain"
python task3_soap_generation.py
```

---

## Implementation Highlights

- spaCy (and optionally scispaCy) for medical NER and phrase extraction.
- DistilBERT for sentiment, mapped to medical categories with rules.
- Rule-based and model-based mapping for SOAP note generation.# Physician Notetaker - Quick Start

Minimal instructions to install dependencies and run each task.

## Install
```bash
pip install -r emitrr/requirements.txt
python -m spacy download en_core_web_sm
# Optional (better clinical coverage):
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

## Run

### Task 1 — Medical NLP Summarization
```python
from task1_medical_ner import process_task1
import json

conversation = """
Physician: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""

print(json.dumps(process_task1(conversation), indent=2))
```

### Task 2 — Sentiment & Intent (CLI)
```bash
python emitrr/task2_sentiment_intent.py --text "I'm worried about my back pain"
```

### Task 3 — SOAP Note Generation
```bash
python emitrr/task3_soap_generation.py
```

## Notes
- Task 1 uses spaCy (and will auto-use scispaCy if installed) with rule-assisted extraction.
- Task 2 uses a Transformer (DistilBERT) plus medical mapping rules; CLI supports `--text` and `--file`.
- Task 3 maps content into SOAP (Subjective, Objective, Assessment, Plan) and prints JSON + formatted text.

## Submission Smoke Tests
```bash
# Task 2 quick check
python emitrr/task2_sentiment_intent.py --text "I'm worried about my back pain"

# Task 3 quick check
python emitrr/task3_soap_generation.py
```

## Implementation Summary (brief)
- spaCy-based medical extraction with optional scispaCy; explicit fallbacks for missing data.
- Temporal parsing for durations/timeframes; prognosis phrasing extraction.
- DistilBERT sentiment → mapped to Anxious/Neutral/Reassured with anxiety-priority + negation handling; regex intents.
- SOAP note assembly from extracted elements; readable formatting + JSON.

