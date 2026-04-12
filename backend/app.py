from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.summarizer import TextSummarizer
from utils.mcq_generator import MCQGenerator
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
try:
    summarizer = TextSummarizer()
    mcq_gen = MCQGenerator()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")

@app.route('/api/generate', methods=['POST'])
def generate_questify():
    """
    Generate summary and MCQs from input text
    Request JSON:
    {
        "text": "input text",
        "num_questions": 5
    }
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data.get('text', '').strip()
        num_questions = int(data.get('num_questions', 5))
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if num_questions < 1 or num_questions > 20:
            return jsonify({'error': 'Number of questions must be between 1 and 20'}), 400
        
        logger.info(f"Processing request: {len(text)} chars, {num_questions} questions")
        
        # Generate summary
        summary = summarizer.summarize(text)
        
        # Generate MCQs
        mcqs = mcq_gen.generate_mcqs(text, num_questions)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'mcqs': mcqs,
            'num_questions': len(mcqs)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in generate_questify: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
