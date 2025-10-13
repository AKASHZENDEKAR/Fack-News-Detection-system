import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class SimpleFakeNewsDetector:
    def __init__(self, vectorizer, true_news_vectors):
        self.vectorizer = vectorizer
        self.true_news_vectors = true_news_vectors
        self.similarity_threshold = 0.3  # Increased threshold for better accuracy
        self.min_length = 50  # Minimum text length for news
    
    def preprocess_text(self, text):
        """Clean and preprocess the input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def is_valid_news_text(self, text):
        """Check if text looks like actual news content"""
        # Check minimum length
        if len(text) < self.min_length:
            return False
        
        # Check for news-like patterns
        news_indicators = [
            r'\b(?:reported|according|sources|officials|government|president|minister|said|told|announced)\b',
            r'\b(?:breaking|news|update|latest|developing)\b',
            r'\b(?:Reuters|AP|CNN|BBC|NBC|CBS|ABC)\b',
            r'\b(?:WASHINGTON|NEW YORK|LONDON|PARIS|TOKYO)\b'
        ]
        
        # If text contains news indicators, it's more likely to be news
        for pattern in news_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # If no news indicators but long enough, still consider it
        return len(text) > 100
    
    def predict(self, text):
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Check if it looks like news content
        if not self.is_valid_news_text(text):
            return 0  # Not news-like content, likely fake/invalid
        
        # Vectorize the input text
        text_vector = self.vectorizer.transform([text])
        
        # Calculate similarity with all true news articles
        similarities = cosine_similarity(text_vector, self.true_news_vectors)
        max_similarity = np.max(similarities)
        
        # If similarity is high, it's likely true news (1), otherwise fake (0)
        return 1 if max_similarity > self.similarity_threshold else 0
