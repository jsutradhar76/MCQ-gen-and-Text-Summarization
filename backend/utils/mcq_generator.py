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
            if not text:
                return []

            # Tokenize and tag
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # Extract nouns and proper nouns (potential answers)
            keywords = [
                word for word, pos in tagged
                if pos in ['NN', 'NNS', 'NNP', 'NNPS']
                and word not in self.stop_words
                and len(word) > 3
                and word.isalpha()
            ]

            # Remove useless/common words
            blacklist = {'thing', 'something', 'anything', 'everything'}
            keywords = [k for k in keywords if k not in blacklist]

            # Count frequency
            keyword_freq = Counter(keywords)

            # Sort by frequency AND word length (better importance)
            sorted_keywords = sorted(
                keyword_freq.items(),
                key=lambda x: (x[1], len(x[0])),
                reverse=True
            )

            top_keywords = [word for word, _ in sorted_keywords[:top_n]]

            return top_keywords

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

            # Get synonyms from WordNet
            synsets = wordnet.synsets(answer_word)

            if synsets:
                for synset in synsets:
                    for lemma in synset.lemmas():
                        distractors.add(lemma.name().replace('_', ' '))

                    for lemma in synset.lemmas():
                        for antonym in lemma.antonyms():
                            distractors.add(antonym.name().replace('_', ' '))

                    for hyponym in synset.hyponyms():
                        for lemma in hyponym.lemmas():
                            distractors.add(lemma.name().replace('_', ' '))

                    for hypernym in synset.hypernyms():
                        for lemma in hypernym.lemmas():
                            distractors.add(lemma.name().replace('_', ' '))

            # 🔥 Clean distractors
            distractors = [
                d for d in distractors
                if d.lower() != answer_word.lower()
                and len(d.split()) <= 3
                and d.isalpha()
            ]

            # 🔥 Remove duplicates
            distractors = list(set(distractors))

            # If enough distractors, return random sample
            if len(distractors) >= num_distractors:
                return random.sample(distractors, num_distractors)

            # 🔥 Context-based fallback (improved)
            context_words = {
                'intelligence': ['ability', 'skill', 'knowledge', 'learning'],
                'learning': ['training', 'education', 'practice', 'study'],
                'model': ['framework', 'structure', 'design', 'pattern'],
                'algorithm': ['method', 'process', 'technique', 'procedure'],
                'network': ['system', 'structure', 'connection', 'framework'],
                'data': ['information', 'input', 'content', 'resource'],
                'feature': ['attribute', 'property', 'quality', 'aspect'],
                'function': ['role', 'purpose', 'operation', 'task']
            }

            for key, values in context_words.items():
                if key in answer_word.lower() or answer_word.lower() in key:
                    for val in values:
                        if val not in distractors and len(distractors) < num_distractors:
                            distractors.append(val)

            # 🔥 Final fallback (clean + relevant)
            if len(distractors) < num_distractors:
                word_pool = [
                    'system', 'process', 'method', 'technique',
                    'concept', 'approach', 'model', 'structure',
                    'function', 'operation', 'mechanism', 'framework'
                ]

                for word in word_pool:
                    if word not in distractors and word.lower() != answer_word.lower():
                        distractors.append(word)
                    if len(distractors) >= num_distractors:
                        break

            return distractors[:num_distractors]

        except Exception as e:
            logger.error(f"Error getting distractors: {e}")
            return ['option1', 'option2', 'option3'][:num_distractors]
    
    def generate_question(self, text, answer):
        try:
            if not text or not answer:
                return f"What is the significance of {answer}?"

            answer = answer.strip()

            sentences = sent_tokenize(text)
            relevant_sentences = [
                s.strip() for s in sentences 
                if answer.lower() in s.lower()
            ]

            if relevant_sentences:
                main_sentence = max(relevant_sentences, key=len)

                # 🔥 Replace answer with blank
                question_sentence = re.sub(
                    rf"\b{re.escape(answer)}\b", 
                    "______", 
                    main_sentence, 
                    flags=re.IGNORECASE
                )

                # 🔥 Clean formatting
                question_sentence = question_sentence.strip()

                # Ensure it ends properly
                if not question_sentence.endswith('?'):
                    question_sentence += " ?"

                return f"Fill in the blank: {question_sentence}"

            else:
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
            used_questions = set()

            # Extract keywords
            keywords = self.extract_keywords(text, top_n=min(num_questions * 2, 20))

            if not keywords:
                logger.warning("No keywords extracted")
                return []

            # Generate MCQs
            for keyword in keywords:
                if len(mcqs) >= num_questions:
                    break

                # Generate question
                question = self.generate_question(text, keyword)

                # Avoid duplicate questions
                if question in used_questions:
                    continue
                used_questions.add(question)

                # Get distractors
                distractors = self.get_distractors(keyword, num_distractors=3)

                # Skip if distractors are weak
                if not distractors or len(distractors) < 3:
                    continue

                # Create options
                options = [keyword] + distractors
                options = list(set(options))  # remove accidental duplicates

                # Ensure exactly 4 options
                if len(options) < 4:
                    continue
                elif len(options) > 4:
                    options = options[:4]

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
        
        except Exception as e:
            logger.error(f"Error in generate_mcqs: {e}")
            return []
