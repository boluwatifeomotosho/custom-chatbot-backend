import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing utilities for the chatbot"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        # Clean text first
        text = self.clean_text(text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)

        return processed_tokens

    def preprocess_for_model(self, text):
        """Preprocess text for model input"""
        tokens = self.tokenize_and_lemmatize(text)
        return ' '.join(tokens)

class ConversationLogger:
    """Log and manage conversation history"""

    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history

    def add_message(self, user_message, bot_response, intent=None, confidence=None):
        """Add a message pair to conversation history"""
        message_data = {
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent,
            'confidence': confidence,
            'timestamp': self._get_timestamp()
        }

        self.conversation_history.append(message_data)

        # Keep only the last max_history messages
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_context(self, num_messages=3):
        """Get recent conversation context"""
        if len(self.conversation_history) < num_messages:
            return self.conversation_history
        return self.conversation_history[-num_messages:]

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

class IntentAnalyzer:
    """Analyze and enhance intent prediction"""

    def __init__(self):
        self.confidence_threshold = 0.7
        self.fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Could you provide more details about what you're looking for?",
            "I'm having trouble understanding. Can you try asking differently?",
            "That's not quite clear to me. Could you elaborate?"
        ]

    def analyze_prediction(self, predictions):
        """Analyze model predictions and determine confidence"""
        if not predictions:
            return None, 0.0

        top_prediction = predictions[0]
        intent = top_prediction['intent']
        confidence = float(top_prediction['probability'])

        return intent, confidence

    def should_use_fallback(self, confidence):
        """Determine if fallback response should be used"""
        return confidence < self.confidence_threshold

    def get_fallback_response(self):
        """Get a random fallback response"""
        import random
        return random.choice(self.fallback_responses)

class ResponseEnhancer:
    """Enhance bot responses with context and personalization"""

    def __init__(self):
        self.user_name = None
        self.conversation_count = 0

    def enhance_response(self, response, user_message, intent=None):
        """Enhance response based on context"""
        self.conversation_count += 1

        # Add personalization if user name is known
        if self.user_name and intent in ['greeting', 'thanks']:
            response = f"{response.rstrip('.')} {self.user_name}!"

        # Add conversation flow enhancements
        if self.conversation_count == 1 and intent == 'greeting':
            response += " Is this your first time chatting with me?"

        return response

    def extract_user_name(self, message):
        """Try to extract user name from message"""
        # Simple name extraction - can be enhanced
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                self.user_name = match.group(1).capitalize()
                return self.user_name

        return None

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]

def validate_input(message):
    """Validate user input"""
    if not message or not message.strip():
        return False, "Message cannot be empty"

    if len(message) > 1000:
        return False, "Message too long"

    # Check for potential spam or inappropriate content
    spam_indicators = ['http://', 'https://', 'www.', '.com', '.org']
    if any(indicator in message.lower() for indicator in spam_indicators):
        return False, "URLs not allowed"

    return True, "Valid"
