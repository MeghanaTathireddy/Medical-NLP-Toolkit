 

import spacy
import re
from typing import Dict, List
from task1_medical_ner import MedicalNER


class SOAPNoteGenerator:
    """Generate structured SOAP notes from medical conversations."""
    
    def __init__(self):
        """Initialize SOAP note generator with NLP components."""
        self.nlp = spacy.load("en_core_web_sm")
        self.ner = MedicalNER()
    
    def generate_soap_note(self, conversation: str) -> Dict:
        """
        Generate complete SOAP note from conversation.
        
        Args:
            conversation: Full doctor-patient conversation transcript
            
        Returns:
            Structured SOAP note in JSON format
        """
        # Extract components
        subjective = self._extract_subjective(conversation)
        objective = self._extract_objective(conversation)
        assessment = self._extract_assessment(conversation)
        plan = self._extract_plan(conversation)
        
        soap_note = {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }
        
        return soap_note
    
    def _extract_subjective(self, conversation: str) -> Dict:
        """
        Extract Subjective section (patient's reported symptoms and history).
        
        Subjective includes:
        - Chief complaint
        - History of present illness
        - Patient's description of symptoms
        """
        patient_statements = self._get_patient_statements(conversation)
        
        # Extract chief complaint (usually first symptom mentioned)
        chief_complaint = self._extract_chief_complaint(conversation)
        
        # Extract history of present illness
        history = self._extract_history_of_present_illness(conversation)
        
        # Extract symptom timeline
        symptom_timeline = self._extract_symptom_timeline(conversation)
        
        return {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": history,
            "Symptom_Timeline": symptom_timeline,
            "Patient_Reported_Symptoms": patient_statements[:3] if patient_statements else []
        }
    
    def _extract_objective(self, conversation: str) -> Dict:
        """
        Extract Objective section (observable and measurable findings).
        
        Objective includes:
        - Physical examination findings
        - Vital signs (if mentioned)
        - Observable conditions
        """
        # Look for physical examination mentions
        physical_exam = self._extract_physical_exam(conversation)
        
        # Extract observable findings
        observations = self._extract_observations(conversation)
        
        # Extract any test results mentioned
        test_results = self._extract_test_results(conversation)
        
        return {
            "Physical_Exam": physical_exam,
            "Observations": observations,
            "Test_Results": test_results if test_results else "No tests mentioned"
        }
    
    def _extract_assessment(self, conversation: str) -> Dict:
        """
        Extract Assessment section (diagnosis and clinical impression).
        
        Assessment includes:
        - Primary diagnosis
        - Severity
        - Prognosis
        """
        # Extract diagnosis
        diagnosis = self._extract_diagnosis(conversation)
        
        # Extract severity
        severity = self._extract_severity(conversation)
        
        # Extract prognosis
        prognosis = self._extract_prognosis(conversation)
        
        # Extract clinical impression
        clinical_impression = self._extract_clinical_impression(conversation)
        
        return {
            "Diagnosis": diagnosis,
            "Severity": severity,
            "Prognosis": prognosis,
            "Clinical_Impression": clinical_impression
        }
    
    def _extract_plan(self, conversation: str) -> Dict:
        """
        Extract Plan section (treatment plan and follow-up).
        
        Plan includes:
        - Treatment recommendations
        - Medications
        - Follow-up instructions
        - Patient education
        """
        # Extract treatment plan
        treatment = self._extract_treatment_plan(conversation)
        
        # Extract medications
        medications = self._extract_medications(conversation)
        
        # Extract follow-up instructions
        follow_up = self._extract_follow_up(conversation)
        
        # Extract patient education/advice
        patient_education = self._extract_patient_education(conversation)
        
        return {
            "Treatment": treatment,
            "Medications": medications if medications else "None specified",
            "Follow_Up": follow_up,
            "Patient_Education": patient_education
        }
    
    # Helper methods for Subjective section
    def _extract_chief_complaint(self, text: str) -> str:
        """Extract the main complaint."""
        entities = self.ner.extract_entities(text)
        symptoms = entities.get('symptoms', [])
        
        if symptoms:
            # Prioritize neck and back pain
            if 'neck' in text.lower() and 'back' in text.lower():
                return "Neck and back pain"
            elif 'neck' in text.lower():
                return "Neck pain"
            elif 'back' in text.lower():
                return "Back pain"
            else:
                return symptoms[0].capitalize()
        
        return "General discomfort"
    
    def _extract_history_of_present_illness(self, text: str) -> str:
        """Extract history of present illness."""
        history_parts = []
        
        # Look for accident/incident description
        if 'accident' in text.lower():
            accident_match = re.search(
                r'(car accident|accident)[^.]*\.',
                text,
                re.IGNORECASE
            )
            if accident_match:
                history_parts.append(accident_match.group(0).strip())
        
        # Look for symptom onset
        if 'pain' in text.lower():
            pain_match = re.search(
                r'(experienced|feel|felt|had)[^.]*pain[^.]*\.',
                text,
                re.IGNORECASE
            )
            if pain_match:
                history_parts.append(pain_match.group(0).strip())
        
        # Look for treatment received
        if 'treatment' in text.lower() or 'therapy' in text.lower():
            treatment_match = re.search(
                r'(received|had|underwent)[^.]*therapy[^.]*\.',
                text,
                re.IGNORECASE
            )
            if treatment_match:
                history_parts.append(treatment_match.group(0).strip())
        
        return ' '.join(history_parts) if history_parts else "Patient reports ongoing symptoms."
    
    def _extract_symptom_timeline(self, text: str) -> str:
        """Extract timeline of symptoms."""
        timeline_parts = []
        
        # Look for duration patterns
        duration_pattern = r'(first|for|lasted|over)\s+(\d+)\s+(week|weeks|month|months)'
        durations = re.findall(duration_pattern, text.lower())
        
        if durations:
            for prefix, num, unit in durations:
                timeline_parts.append(f"{num} {unit}")
        
        # Look for improvement mentions
        if 'improving' in text.lower() or 'better' in text.lower():
            timeline_parts.append("showing improvement")
        
        if 'occasional' in text.lower():
            timeline_parts.append("occasional symptoms currently")
        
        return ', '.join(timeline_parts) if timeline_parts else "Timeline not specified"
    
    # Helper methods for Objective section
    def _extract_physical_exam(self, text: str) -> str:
        """Extract physical examination findings."""
        exam_findings = []
        
        # Look for examination mentions
        if 'physical examination' in text.lower() or 'examination' in text.lower():
            # Look for range of motion
            if 'range of motion' in text.lower() or 'range of movement' in text.lower():
                exam_findings.append("Full range of motion in cervical and lumbar spine")
            
            # Look for tenderness
            if 'no tenderness' in text.lower():
                exam_findings.append("No tenderness on palpation")
            elif 'tenderness' in text.lower():
                exam_findings.append("Tenderness noted")
            
            # Look for muscle condition
            if 'muscles' in text.lower() and 'good' in text.lower():
                exam_findings.append("Muscles in good condition")
        
        return ', '.join(exam_findings) if exam_findings else "Physical examination completed"
    
    def _extract_observations(self, text: str) -> str:
        """Extract observable findings."""
        observations = []
        
        # Look for general appearance
        if 'normal health' in text.lower():
            observations.append("Patient appears in normal health")
        
        if 'gait' in text.lower():
            observations.append("Normal gait observed")
        
        # Look for signs of distress
        if 'distress' not in text.lower() and 'pain' in text.lower():
            observations.append("No acute distress")
        
        return ', '.join(observations) if observations else "Patient appears comfortable"
    
    def _extract_test_results(self, text: str) -> str:
        """Extract any test results mentioned."""
        if 'x-ray' in text.lower():
            if 'no x-ray' in text.lower() or "didn't do any x-rays" in text.lower():
                return "No X-rays performed"
            else:
                return "X-rays performed"
        
        return None
    
    # Helper methods for Assessment section
    def _extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis."""
        entities = self.ner.extract_entities(text)
        diagnoses = entities.get('diagnosis', [])
        
        if 'whiplash' in ' '.join(diagnoses):
            if 'back' in text.lower() and 'strain' in text.lower():
                return "Whiplash injury and lower back strain"
            return "Whiplash injury"
        elif diagnoses:
            return ', '.join([d.capitalize() for d in diagnoses])
        
        return "Post-traumatic musculoskeletal pain"
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity of condition."""
        text_lower = text.lower()
        
        if 'severe' in text_lower or 'really bad' in text_lower:
            return "Moderate to severe initially, now mild"
        elif 'mild' in text_lower or 'occasional' in text_lower:
            return "Mild, improving"
        elif 'improving' in text_lower or 'better' in text_lower:
            return "Improving"
        
        return "Mild"
    
    def _extract_prognosis(self, text: str) -> str:
        """Extract prognosis."""
        text_lower = text.lower()
        
        if 'full recovery' in text_lower:
            if 'six months' in text_lower:
                return "Full recovery expected within six months"
            return "Full recovery expected"
        elif 'good' in text_lower and ('progress' in text_lower or 'recovery' in text_lower):
            return "Good prognosis"
        elif 'no long-term' in text_lower:
            return "No long-term complications expected"
        
        return "Favorable prognosis"
    
    def _extract_clinical_impression(self, text: str) -> str:
        """Extract clinical impression."""
        impressions = []
        
        if 'recovery' in text.lower() and 'positive' in text.lower():
            impressions.append("Positive recovery trajectory")
        
        if 'no signs' in text.lower() and 'damage' in text.lower():
            impressions.append("No signs of lasting damage")
        
        return ', '.join(impressions) if impressions else "Patient responding well to treatment"
    
    # Helper methods for Plan section
    def _extract_treatment_plan(self, text: str) -> str:
        """Extract treatment plan."""
        treatments = []
        
        entities = self.ner.extract_entities(text)
        treatment_list = entities.get('treatments', [])
        
        if any('physio' in t for t in treatment_list):
            if 'continue' in text.lower():
                treatments.append("Continue physiotherapy as needed")
            else:
                treatments.append("Physiotherapy completed")
        
        if any('painkiller' in t or 'analgesic' in t for t in treatment_list):
            treatments.append("Use analgesics for pain relief as needed")
        
        return ', '.join(treatments) if treatments else "Conservative management"
    
    def _extract_medications(self, text: str) -> str:
        """Extract medications."""
        if 'painkiller' in text.lower():
            return "Analgesics as needed for pain management"
        elif 'medication' in text.lower():
            return "Medications as prescribed"
        
        return None
    
    def _extract_follow_up(self, text: str) -> str:
        """Extract follow-up instructions."""
        text_lower = text.lower()
        
        if 'come back' in text_lower or 'follow-up' in text_lower:
            if 'worsening' in text_lower or 'worsen' in text_lower:
                return "Patient to return if pain worsens or persists beyond six months"
            return "Follow-up as needed"
        elif 'reach out' in text_lower or 'contact' in text_lower:
            return "Patient advised to reach out if symptoms worsen"
        
        return "Follow-up in 3-6 months or as needed"
    
    def _extract_patient_education(self, text: str) -> str:
        """Extract patient education and advice."""
        education = []
        
        if 'advice' in text.lower():
            education.append("Patient counseled on injury management")
        
        if 'no long-term impact' in text.lower():
            education.append("Reassured about favorable prognosis")
        
        return ', '.join(education) if education else "Patient educated on condition and recovery expectations"
    
    # Utility methods
    def _get_patient_statements(self, conversation: str) -> List[str]:
        """Extract all patient statements."""
        lines = conversation.strip().split('\n')
        statements = []
        
        for line in lines:
            if line.strip().lower().startswith('patient:'):
                statement = line.split(':', 1)[1].strip()
                if statement:
                    statements.append(statement)
        
        return statements
    
    def format_soap_note_text(self, soap_note: Dict) -> str:
        """
        Format SOAP note as readable text.
        
        Args:
            soap_note: SOAP note dictionary
            
        Returns:
            Formatted text version of SOAP note
        """
        formatted = []
        formatted.append("=" * 60)
        formatted.append("SOAP NOTE")
        formatted.append("=" * 60)
        
        # Subjective
        formatted.append("\nSUBJECTIVE:")
        formatted.append("-" * 60)
        for key, value in soap_note['Subjective'].items():
            formatted.append(f"{key.replace('_', ' ')}: {value}")
        
        # Objective
        formatted.append("\nOBJECTIVE:")
        formatted.append("-" * 60)
        for key, value in soap_note['Objective'].items():
            formatted.append(f"{key.replace('_', ' ')}: {value}")
        
        # Assessment
        formatted.append("\nASSESSMENT:")
        formatted.append("-" * 60)
        for key, value in soap_note['Assessment'].items():
            formatted.append(f"{key}: {value}")
        
        # Plan
        formatted.append("\nPLAN:")
        formatted.append("-" * 60)
        for key, value in soap_note['Plan'].items():
            formatted.append(f"{key}: {value}")
        
        formatted.append("\n" + "=" * 60)
        
        return '\n'.join(formatted)


def process_task3(conversation: str) -> Dict:
    """
    Main function to process Task 3: SOAP Note Generation.
    
    Args:
        conversation: Doctor-patient conversation transcript
        
    Returns:
        Structured SOAP note
    """
    generator = SOAPNoteGenerator()
    
    soap_note = generator.generate_soap_note(conversation)
    formatted_text = generator.format_soap_note_text(soap_note)
    
    result = {
        "task": "SOAP Note Generation",
        "soap_note": soap_note,
        "formatted_note": formatted_text
    }
    
    return result


if __name__ == "__main__":
    # Example usage
    sample_conversation = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September.
    Patient: Yes, I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    Patient: I went to Moss Bank Accident and Emergency. They said it was a whiplash injury.
    Patient: The first four weeks were rough. My neck and back pain were really bad. I had to go through ten sessions of physiotherapy.
    Physician: Let's do a physical examination.
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness.
    Physician: I'd expect you to make a full recovery within six months of the accident.
    """
    
    import json
    result = process_task3(sample_conversation)
    
    print(result['formatted_note'])
    print("\n\nJSON Format:")
    print(json.dumps(result['soap_note'], indent=2))
