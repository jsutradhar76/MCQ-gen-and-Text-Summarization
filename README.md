# Questify - Text Summarization & MCQ Generator

Simple web app that summarizes text and generates multiple-choice questions.

## Setup (2 minutes)

### Step 1: Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Start the Backend
```bash
cd backend
python app.py
```
The server will run on `http://localhost:5000`

### Step 3: Open the Frontend
Open this file in your browser:
```
frontend/index.html
```

That's it! The app is ready to use.

## How to Use

1. Paste text into the textarea (at least 50 words recommended)
2. Select number of MCQs you want (1-20)
3. Click "Generate Summary & MCQs"
4. View the summary and multiple choice questions
5. Download results if needed

## File Structure

```
backend/
  ├── app.py                 # Flask API server
  ├── requirements.txt        # Python packages to install
  ├── config.py              # Settings
  └── utils/
      ├── summarizer.py      # Summarization logic
      └── mcq_generator.py    # MCQ generation logic

frontend/
  ├── index.html             # Main page
  ├── styles.css             # Styling
  └── script.js              # JavaScript logic
```

## How It Works

- **Summarization**: Extracts key sentences from the text
- **Keyword Extraction**: Finds important words using NLP
- **Question Generation**: Creates fill-in-the-blank questions
- **Distractors**: Uses WordNet to generate wrong answer options

## Troubleshooting

**Port 5000 already in use?**
Edit `backend/app.py` line 73 and change `port=5000` to another port like `port=5001`

**Frontend can't connect to backend?**
Make sure backend is running: `python backend/app.py`

**No MCQs generated?**
Use longer text with more nouns and detailed content
