 

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Tuple
import re


class SentimentIntentAnalyzer:
    """Analyze patient sentiment and intent from medical conversations."""
    
    def __init__(self):
        """Initialize sentiment analysis and intent detection models."""
        # Load pre-trained sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Define intent patterns
        self._setup_intent_patterns()
    
    def _setup_intent_patterns(self):
        """Define patterns for intent detection."""
        self.intent_patterns = {
            'Seeking reassurance': [
                r'will (it|this|i) (get better|be okay|recover)',
                r'(worried|concern|anxious|nervous) about',
                r'should i (worry|be concerned)',
                r'is (it|this) (normal|serious)',
                r'will (it|this) affect me',
                r'do i need to worry'
            ],
            'Reporting symptoms': [
                r'(i have|i feel|i\'m experiencing|i get)',
                r'(pain|hurt|ache|discomfort|stiffness)',
                r'my (neck|back|head|spine) (hurts|aches)',
                r'trouble (sleeping|moving|walking)',
                r'it (hurts|aches) when'
            ],
            'Expressing concern': [
                r'(worried|concerned|anxious|nervous|scared)',
                r'what if',
                r'i\'m afraid',
                r'bothers me',
                r'makes me (worry|nervous)'
            ],
            'Seeking advice': [
                r'what (should|can|could) i',
                r'how (do|can|should) i',
                r'is there anything i (can|should)',
                r'what do you (recommend|suggest|advise)',
                r'should i (take|do|avoid)'
            ],
            'Expressing gratitude': [
                r'thank you',
                r'thanks',
                r'appreciate',
                r'grateful',
                r'helpful'
            ],
            'Describing improvement': [
                r'(better|improving|getting better)',
                r'not as (bad|painful)',
                r'less (pain|discomfort)',
                r'starting to (feel better|improve)'
            ],
            'Describing history': [
                r'(it was|it happened|i was)',
                r'(last|on) (week|month|year|september|january)',
                r'i went to',
                r'they (told|said|gave) me'
            ]
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of patient's text.
        
        Args:
            text: Patient's dialogue text
            
        Returns:
            Dictionary with sentiment classification and confidence score
        """
        # Handle empty or very short text
        if not text or len(text.strip()) < 3:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "raw_label": "NEUTRAL",
                "raw_score": 0.0
            }
        
        # Get sentiment from model
        result = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
        
        # Map to medical context
        sentiment = self._map_to_medical_sentiment(result['label'], result['score'], text)
        
        return {
            "sentiment": sentiment,
            "confidence": round(result['score'], 3),
            "raw_label": result['label'],
            "raw_score": round(result['score'], 3)
        }
    
    def _map_to_medical_sentiment(self, label: str, score: float, text: str) -> str:
        """
        Map generic sentiment to medical context.
        
        Args:
            label: Sentiment label from model (POSITIVE/NEGATIVE)
            score: Confidence score
            text: Original text for context
            
        Returns:
            Medical sentiment category
        """
        text_lower = text.lower()
        
        # Helper to match whole words with boundaries
        def _has_word(word_list, haystack):
            return any(re.search(rf"\b{re.escape(w)}\b", haystack) for w in word_list)

        # Check for anxiety indicators (whole-word)
        anxiety_words = ['worried', 'concern', 'concerned', 'anxious', 'nervous', 'scared', 'afraid']
        has_anxiety = _has_word(anxiety_words, text_lower)
        
        # Handle simple negations that flip anxiety meaning (e.g., "not worried", "no longer worried")
        negated_anxiety_patterns = [
            r"not (worried|concerned|anxious|nervous|scared|afraid)",
            r"no longer (worried|concerned|anxious|nervous|scared|afraid)",
            r"hardly (worried|concerned|anxious|nervous|scared|afraid)",
            r"doesn't make me (worried|concerned|anxious|nervous|scared|afraid)",
            r"does not make me (worried|concerned|anxious|nervous|scared|afraid)"
        ]
        has_negation = any(re.search(p, text_lower) for p in negated_anxiety_patterns)
        if has_negation:
            has_anxiety = False
        
        # Check for reassurance indicators
        reassurance_words = ['better', 'good', 'relief', 'thank', 'appreciate', 'helpful', 'manageable', 'under control']
        has_reassurance = _has_word(reassurance_words, text_lower)

        # Additional recovery/normalcy cues that imply reassurance even with some negative phrasing
        recovery_phrases = [
            'back to my usual routine',
            'back to my routine',
            'back to normal',
            'returned to normal',
            "hasn't really stopped me",
            "hasn't stopped me",
            "didn't really stop me",
            "didn't stop me",
            'no longer',
            'able to do everything',
            'doing everything as usual',
            'recovered',
            'improved',
            'improving',
            'getting better',
            'better now',
            'feels fine now',
            'okay now',
            'normal now',
            'back at work',
            'back to work'
        ]
        has_recovery = any(phrase in text_lower for phrase in recovery_phrases)
        
        # Apply revised decision rule:
        # 1) If explicit anxiety and no negation and no strong recovery/reassurance → Anxious
        if has_anxiety and not has_negation and not (has_reassurance or has_recovery):
            return "Anxious"
        # 2) Else if strong recovery/reassurance → Reassured
        if has_recovery or has_reassurance:
            return "Reassured"
        # 3) Else fallback to model/Neutral (guard extremes)
        if label == 'NEGATIVE' and score > 0.9:
            return "Anxious"
        if label == 'POSITIVE' and score > 0.8:
            return "Reassured"
        return "Neutral"
    
    def detect_intent(self, text: str) -> str:
        """
        Detect patient's intent from text.
        
        Args:
            text: Patient's dialogue text
            
        Returns:
            Detected intent category
        """
        text_lower = text.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Get the intent with highest score
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        return "General conversation"
    
    def detect_multiple_intents(self, text: str) -> List[str]:
        """
        Detect multiple intents in text.
        
        Args:
            text: Patient's dialogue text
            
        Returns:
            List of detected intents
        """
        text_lower = text.lower()
        detected_intents = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_intents.append(intent)
                    break
        
        return detected_intents if detected_intents else ["General conversation"]
    
    def analyze_patient_dialogue(self, conversation: str) -> List[Dict]:
        """
        Analyze each patient statement in the conversation.
        
        Args:
            conversation: Full conversation transcript
            
        Returns:
            List of analysis results for each patient statement
        """
        # Extract patient statements
        patient_statements = self._extract_patient_statements(conversation)
        
        results = []
        for statement in patient_statements:
            sentiment = self.analyze_sentiment(statement)
            intent = self.detect_intent(statement)
            
            results.append({
                "statement": statement,
                "sentiment": sentiment['sentiment'],
                "confidence": sentiment['confidence'],
                "intent": intent
            })
        
        return results
    
    def _extract_patient_statements(self, conversation: str) -> List[str]:
        """Extract patient statements from conversation."""
        lines = conversation.strip().split('\n')
        patient_statements = []
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('patient:'):
                # Remove "Patient:" prefix
                statement = line.split(':', 1)[1].strip()
                if statement:
                    patient_statements.append(statement)
        
        return patient_statements
    
    def generate_overall_sentiment(self, conversation: str) -> Dict:
        """
        Generate overall sentiment analysis for the entire conversation.
        
        Args:
            conversation: Full conversation transcript
            
        Returns:
            Overall sentiment summary
        """
        patient_analyses = self.analyze_patient_dialogue(conversation)
        
        if not patient_analyses:
            return {
                "overall_sentiment": "Neutral",
                "sentiment_distribution": {},
                "dominant_intent": "General conversation",
                "confidence": 0.0
            }
        
        # Calculate sentiment distribution
        sentiment_counts = {}
        intent_counts = {}
        total_confidence = 0
        
        for analysis in patient_analyses:
            sentiment = analysis['sentiment']
            intent = analysis['intent']
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            total_confidence += analysis['confidence']
        
        # Determine overall sentiment
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        dominant_intent = max(intent_counts, key=intent_counts.get)
        avg_confidence = total_confidence / len(patient_analyses)
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts,
            "dominant_intent": dominant_intent,
            "intent_distribution": intent_counts,
            "average_confidence": round(avg_confidence, 3),
            "statement_count": len(patient_analyses)
        }


def process_task2(statement: str) -> Dict:
    """
    Main function to process Task 2: Sentiment & Intent Analysis.
    
    Implements both deliverables:
    1. Sentiment Classification using Transformer (DistilBERT)
    2. Intent Detection
    
    Args:
        statement: Single patient statement/dialogue
        
    Returns:
        Expected Output (JSON):
        {
          "Sentiment": "Anxious",
          "Intent": "Seeking reassurance"
        }
    """
    analyzer = SentimentIntentAnalyzer()
    
    sentiment = analyzer.analyze_sentiment(statement)
    intent = analyzer.detect_intent(statement)
    
    return {
        "Sentiment": sentiment['sentiment'],
        "Intent": intent
    }


if __name__ == "__main__":
    import json
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Task 2: Sentiment & Intent Analysis")
    parser.add_argument("--text", type=str, help="Patient dialogue text to analyze")
    parser.add_argument("--file", type=str, help="Path to a text file containing the dialogue")
    args = parser.parse_args()

    # Resolve input source priority: --text > --file > default demo
    if args.text:
        statement = args.text
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                statement = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        # Default demo input
        statement = "I'm a bit worried about my back pain, but I hope it gets better soon."

    print("=" * 80)
    print("TASK 2: SENTIMENT & INTENT ANALYSIS")
    print("=" * 80)
    print("\nSample Input (Patient's Dialogue):")
    print(f'"{statement}"')
    print("\n" + "=" * 80)
    print("Output (JSON):")
    print("=" * 80)
    
    result = process_task2(statement)
    print(json.dumps(result, indent=2))
