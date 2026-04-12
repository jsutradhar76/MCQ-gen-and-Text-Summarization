import re
import random
import logging
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class MCQGenerator:
    def __init__(self):
        """Initialize MCQ generator"""
        self.stop_words = set(stopwords.words('english'))
        logger.info("MCQ Generator initialized")
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract important keywords/entities from text
        
        Args:
            text (str): Input text
            top_n (int): Number of keywords to extract
        
        Returns:
            list: List of keywords
        """
        try:
            # Tokenize and tag
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # Extract nouns and proper nouns (potential answers)
            keywords = [word for word, pos in tagged 
                       if pos in ['NN', 'NNS', 'NNP', 'NNPS'] 
                       and word not in self.stop_words
                       and len(word) > 3]
            
            # Count frequency and get top keywords
            keyword_freq = Counter(keywords)
            top_keywords = [word for word, freq in keyword_freq.most_common(top_n)]
            
            return top_keywords[:top_n]
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def get_distractors(self, answer_word, num_distractors=3):
        """
        Get distractor words using WordNet synonyms, antonyms, and related concepts
        
        Args:
            answer_word (str): The correct answer
            num_distractors (int): Number of distractors needed
        
        Returns:
            list: List of distractor words
        """
        try:
            distractors = set()
            
            # Get synonyms from wordnet
            synsets = wordnet.synsets(answer_word)
            
            if synsets:
                # Get synonyms
                for synset in synsets:
                    for lemma in synset.lemmas():
                        if lemma.name() != answer_word:
                            distractors.add(lemma.name().replace('_', ' '))
                
                # Get antonyms (opposite meanings)
                for synset in synsets:
                    for lemma in synset.lemmas():
                        for antonym in lemma.antonyms():
                            if antonym.name() != answer_word:
                                distractors.add(antonym.name().replace('_', ' '))
                
                # Get hyponyms (more specific concepts)
                for synset in synsets:
                    for hyponym in synset.hyponyms():
                        for lemma in hyponym.lemmas():
                            if lemma.name() != answer_word:
                                distractors.add(lemma.name().replace('_', ' '))
                
                # Get hypernyms (broader categories)
                for synset in synsets:
                    for hypernym in synset.hypernyms():
                        for lemma in hypernym.lemmas():
                            if lemma.name() != answer_word:
                                distractors.add(lemma.name().replace('_', ' '))
            
            # Convert to list and remove answer_word and close variants
            distractors = list(distractors)
            distractors = [d for d in distractors if d.lower() != answer_word.lower()]
            
            # If we have enough quality distractors, return them
            if len(distractors) >= num_distractors:
                return random.sample(distractors, num_distractors)
            
            # If still not enough, add contextual alternatives based on word type
            context_words = {
                'intelligence': ['competence', 'capability', 'ability', 'skill', 'knowledge', 'learning'],
                'learning': ['training', 'education', 'practice', 'study', 'experience', 'development'],
                'model': ['framework', 'design', 'structure', 'architecture', 'template', 'pattern'],
                'algorithm': ['method', 'procedure', 'technique', 'strategy', 'approach', 'mechanism'],
                'network': ['connection', 'link', 'structure', 'system', 'web', 'infrastructure'],
                'data': ['information', 'content', 'input', 'material', 'resource', 'evidence'],
                'feature': ['characteristic', 'attribute', 'property', 'quality', 'aspect', 'element'],
                'function': ['role', 'purpose', 'task', 'operation', 'action', 'duty']
            }
            
            # Try to find contextually relevant distractors
            for key, values in context_words.items():
                if key.lower() in answer_word.lower() or answer_word.lower() in key:
                    for val in values:
                        if len(distractors) < num_distractors:
                            distractors.append(val)
            
            # Fill remaining with random related words of similar length
            if len(distractors) < num_distractors:
                length = len(answer_word)
                word_pool = ['ability', 'process', 'method', 'technique', 'concept', 'system', 
                            'component', 'element', 'aspect', 'quality', 'property', 'nature',
                            'structure', 'pattern', 'framework', 'approach', 'strategy', 'tool']
                
                for word in random.sample(word_pool, min(len(word_pool), num_distractors - len(distractors))):
                    if word not in distractors and word.lower() != answer_word.lower():
                        distractors.append(word)
            
            return distractors[:num_distractors]
        
        except Exception as e:
            logger.error(f"Error getting distractors: {e}")
            return ['alternative1', 'alternative2', 'alternative3'][:num_distractors]
    
    def generate_question(self, text, answer):
        """
        Generate a proper MCQ question from text given an answer
        
        Args:
            text (str): The source text
            answer (str): The answer word
        
        Returns:
            str: Generated question
        """
        try:
            # Basic validation
            if not text or not answer:
                return f"What is the significance of {answer}?"

            answer = answer.strip()
            
            # Find sentences containing the answer
            sentences = sent_tokenize(text)
            relevant_sentences = [
                s.strip() for s in sentences 
                if answer.lower() in s.lower()
            ]
            
            if relevant_sentences:
                # Choose the most informative sentence (longest one)
                main_sentence = max(relevant_sentences, key=len)

                # Question templates
                question_templates = [
                    f"What is mentioned about {answer}?",
                    f"Which of the following best describes {answer}?",
                    f"According to the text, what is {answer}?",
                    f"What role does {answer} play in this context?",
                    f"How is {answer} characterized in the passage?",
                    f"What is the primary characteristic of {answer}?",
                    f"In the context provided, {answer} is best described as:",
                    f"What can be inferred about {answer}?"
                ]

                # Reduce repetition by biasing toward contextual questions
                if len(main_sentence.split()) > 10:
                    preferred_templates = question_templates[2:6]
                    question = random.choice(preferred_templates)
                else:
                    question = random.choice(question_templates)

                return question
            
            else:
                # Smarter fallback
                return f"Which statement is true about {answer}?"

        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return f"What is the significance of {answer}?"
    
    def generate_mcqs(self, text, num_questions=5):
        """
        Generate MCQs from text
        
        Args:
            text (str): Input text
            num_questions (int): Number of MCQs to generate
        
        Returns:
            list: List of MCQ dictionaries with question, options, and correct answer
        """
        try:
            mcqs = []
            
            # Extract keywords
            keywords = self.extract_keywords(text, top_n=min(num_questions * 2, 20))
            
            if not keywords:
                logger.warning("No keywords extracted")
                return []
            
            # Generate MCQs for each keyword
            for keyword in keywords[:num_questions]:
                # Get distractors
                distractors = self.get_distractors(keyword, num_distractors=3)
                
                # Generate question
                question = self.generate_question(text, keyword)
                
                # Create options
                options = [keyword] + distractors
                random.shuffle(options)
                
                # Find correct answer index
                correct_idx = options.index(keyword)
                
                mcq = {
                    'question': question,
                    'options': options,
                    'correct_answer': keyword,
                    'correct_index': correct_idx
                }
                
                mcqs.append(mcq)
            
            return mcqs
        
        except Exception as e:
            logger.error(f"Error in generate_mcqs: {e}")
            return []
