/* ==================== */
/* Questify JavaScript */
/* ==================== */

// Configuration
const API_BASE_URL = 'http://localhost:5000';
const API_ENDPOINT = `${API_BASE_URL}/api/generate`;

// DOM Elements
const questifyForm = document.getElementById('questifyForm');
const textInput = document.getElementById('textInput');
const numQuestionsInput = document.getElementById('numQuestions');
const submitBtn = document.getElementById('submitBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('resultsSection');
const errorContainer = document.getElementById('errorContainer');
const errorMessage = document.getElementById('errorMessage');
const closeErrorBtn = document.getElementById('closeErrorBtn');
const charCount = document.getElementById('charCount');
const copySummaryBtn = document.getElementById('copySummaryBtn');
const resetBtn = document.getElementById('resetBtn');
const downloadBtn = document.getElementById('downloadBtn');
const summaryContent = document.getElementById('summaryContent');
const mcqsContainer = document.getElementById('mcqsContainer');
const inputSection = document.querySelector('.input-section');

// State
let currentSummary = '';
let currentMCQs = [];

/* ==================== */
/* Event Listeners */
/* ==================== */

// Form submission
questifyForm.addEventListener('submit', handleFormSubmit);

// Character count
textInput.addEventListener('input', () => {
    charCount.textContent = textInput.value.length;
});

// Quick select buttons
document.querySelectorAll('.quick-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        const value = btn.dataset.value;
        numQuestionsInput.value = value;
        updateQuickButtonsUI();
    });
});

// Copy summary button
copySummaryBtn.addEventListener('click', copySummaryToClipboard);

// Reset button
resetBtn.addEventListener('click', resetForm);

// Download button
downloadBtn.addEventListener('click', downloadResults);

// Close error button
closeErrorBtn.addEventListener('click', () => {
    errorContainer.style.display = 'none';
});

// Close error by clicking outside
errorContainer.addEventListener('click', (e) => {
    if (e.target === errorContainer) {
        errorContainer.style.display = 'none';
    }
});

/* ==================== */
/* Main Functions */
/* ==================== */

async function handleFormSubmit(e) {
    e.preventDefault();

    const text = textInput.value.trim();
    const numQuestions = parseInt(numQuestionsInput.value);

    // Validation
    if (!validateInput(text, numQuestions)) {
        return;
    }

    // Show loading state
    showLoadingState();

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                num_questions: numQuestions
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to process request');
        }

        const data = await response.json();

        if (data.success) {
            currentSummary = data.summary;
            currentMCQs = data.mcqs;
            displayResults(data);
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoadingState();
    }
}

function validateInput(text, numQuestions) {
    // Check if text is empty
    if (!text) {
        showError('Please enter some text to summarize.');
        return false;
    }

    // Check minimum length
    const wordCount = text.trim().split(/\s+/).length;
    if (wordCount < 20) {
        showError(`Please enter at least 20 words. (Current: ${wordCount} words)`);
        return false;
    }

    // Check number of questions
    if (numQuestions < 1 || numQuestions > 20) {
        showError('Number of questions must be between 1 and 20.');
        return false;
    }

    return true;
}

function displayResults(data) {
    // Display summary
    summaryContent.textContent = data.summary;

    // Display MCQs
    displayMCQs(data.mcqs);

    // Show results section and scroll to it
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

function displayMCQs(mcqs) {
    mcqsContainer.innerHTML = '';

    if (!mcqs || mcqs.length === 0) {
        mcqsContainer.innerHTML = '<p class="loading-text">No MCQs could be generated. Please try with different text.</p>';
        return;
    }

    mcqs.forEach((mcq, index) => {
        const mcqElement = createMCQElement(mcq, index + 1);
        mcqsContainer.appendChild(mcqElement);
    });
}

function createMCQElement(mcq, questionNumber) {
    const mcqDiv = document.createElement('div');
    mcqDiv.className = 'mcq-item';
    mcqDiv.innerHTML = `
        <div class="mcq-number">${questionNumber}</div>
        <div class="mcq-question">${escapeHtml(mcq.question)}</div>
        <div class="mcq-options">
            ${mcq.options.map((option, idx) => `
                <label class="mcq-option">
                    <input 
                        type="radio" 
                        name="mcq-${questionNumber}" 
                        value="${option}"
                        data-correct="${option === mcq.correct_answer}"
                    >
                    <span class="mcq-option-text">${escapeHtml(option)}</span>
                </label>
            `).join('')}
        </div>
        <div class="mcq-answer" style="display: none;">
            <strong>Correct Answer:</strong> ${escapeHtml(mcq.correct_answer)}
        </div>
    `;

    // Add event listeners to radio buttons
    const radios = mcqDiv.querySelectorAll('input[type="radio"]');
    const answerDiv = mcqDiv.querySelector('.mcq-answer');

    radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            // Remove previous styles
            mcqDiv.querySelectorAll('.mcq-option').forEach(opt => {
                opt.classList.remove('correct', 'incorrect');
            });

            // Add style to selected option
            const selectedLabel = e.target.parentElement;
            if (e.target.dataset.correct === 'true') {
                selectedLabel.classList.add('correct');
            } else {
                selectedLabel.classList.add('incorrect');
            }

            // Show answer
            answerDiv.style.display = 'block';
        });
    });

    return mcqDiv;
}

function copySummaryToClipboard() {
    navigator.clipboard.writeText(currentSummary).then(() => {
        const originalText = copySummaryBtn.textContent;
        copySummaryBtn.textContent = '✓ Copied!';
        setTimeout(() => {
            copySummaryBtn.textContent = originalText;
        }, 2000);
    }).catch(() => {
        showError('Failed to copy summary to clipboard.');
    });
}

function resetForm() {
    textInput.value = '';
    numQuestionsInput.value = '5';
    charCount.textContent = '0';
    resultsSection.style.display = 'none';
    currentSummary = '';
    currentMCQs = [];
    updateQuickButtonsUI();
    inputSection.scrollIntoView({ behavior: 'smooth' });
}

function downloadResults() {
    const content = generateDownloadContent();
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(content));
    element.setAttribute('download', 'questify-results.txt');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

function generateDownloadContent() {
    let content = '=== QUESTIFY RESULTS ===\n\n';
    content += '=== SUMMARY ===\n';
    content += currentSummary + '\n\n';
    content += '=== MCQs ===\n\n';

    currentMCQs.forEach((mcq, index) => {
        content += `Question ${index + 1}: ${mcq.question}\n`;
        mcq.options.forEach((option, idx) => {
            const label = String.fromCharCode(65 + idx); // A, B, C, D
            content += `${label}. ${option}\n`;
        });
        content += `Correct Answer: ${mcq.correct_answer}\n\n`;
    });

    return content;
}

/* ==================== */
/* UI Helpers */
/* ==================== */

function showLoadingState() {
    submitBtn.disabled = true;
    loadingContainer.style.display = 'flex';
    questifyForm.style.pointerEvents = 'none';
    questifyForm.style.opacity = '0.6';
}

function hideLoadingState() {
    submitBtn.disabled = false;
    loadingContainer.style.display = 'none';
    questifyForm.style.pointerEvents = 'auto';
    questifyForm.style.opacity = '1';
}

function showError(message) {
    errorMessage.textContent = message;
    errorContainer.style.display = 'flex';
}

function updateQuickButtonsUI() {
    const currentValue = numQuestionsInput.value;
    document.querySelectorAll('.quick-btn').forEach(btn => {
        if (btn.dataset.value === currentValue) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/* ==================== */
/* Initialize */
/* ==================== */

document.addEventListener('DOMContentLoaded', () => {
    updateQuickButtonsUI();
});
