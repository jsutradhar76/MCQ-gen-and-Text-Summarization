import logging
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        """Initialize text summarizer using extractive summarization"""
        self.stop_words = set(stopwords.words('english'))
        logger.info("Summarizer initialized")
    
    def summarize(self, text, num_sentences=3):
        """
        Summarize text using extractive summarization
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in summary
        
        Returns:
            str: Summarized text
        """
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Simple extractive summary: take first N sentences
            summary_sentences = sentences[:num_sentences]
            summary = ' '.join(summary_sentences)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error in summarize: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')
            return '. '.join(sentences[:min(3, len(sentences))]) + '.'
