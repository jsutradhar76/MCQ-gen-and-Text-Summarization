

function displayMCQs(mcqs) {
  const mcqList = document.getElementById("mcqList");
  mcqList.innerHTML = "";

  mcqs.forEach((mcq) => {
      const mcqItem = document.createElement("div");
      mcqItem.className = "mcq-item";

      const question = document.createElement("h3");
      question.textContent = mcq.question;
      mcqItem.appendChild(question);

      const optionsList = document.createElement("ul");
      mcq.options.forEach((option) => {
          const optionItem = document.createElement("li");
          optionItem.textContent = option;
          optionsList.appendChild(optionItem);
      });
      mcqItem.appendChild(optionsList);

      const correctAnswer = document.createElement("p");
      correctAnswer.className = "correct-answer";
      correctAnswer.textContent = `Correct Answer: ${mcq.correctAnswer}`;
      mcqItem.appendChild(correctAnswer);

      mcqList.appendChild(mcqItem);
  });
}



document.getElementById("mcqForm").addEventListener("submit", async function (event) {
  event.preventDefault();

  let context = document.getElementById("textInput").value;
  let method = document.getElementById("methodSelect").value;
  let mcqList = document.getElementById("mcqList");

  if (context.trim() === "") {
      alert("Please enter some text!");
      return;
  }

  mcqList.innerHTML = "<p>Generating MCQs... Please wait.</p>";

  try {
    let response = await fetch("http://localhost:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context: context, method: method })
  });

      let data = await response.json();
      mcqList.innerHTML = `<h3>Summarized Text:</h3><p>${data.summary}</p><hr>${data.mcq}`;
  } catch (error) {
      console.error("Error generating MCQs:", error);
      mcqList.innerHTML = "<p style='color: red;'>Error generating MCQs. Please try again.</p>";
  }
});



// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function() {
  const form = document.getElementById("mcqForm");
  const loader = document.getElementById("loader");
  const mcqList = document.getElementById("mcqList");

  form.addEventListener("submit", function(event) {
    event.preventDefault();  // Prevent form submission (page reload)
    
    // Show the loader
    loader.style.display = "block";

    // Simulate a delay for MCQ generation (replace this with your actual API call or logic)
    setTimeout(function() {
      // Hide the loader after generating MCQs
      loader.style.display = "none";

      // Example: Displaying some dummy MCQs
      mcqList.innerHTML = `
        <div class="mcq-item">
          <p><strong>What is the capital of France?</strong></p>
          <ul>
            <li>Berlin</li>
            <li>Madrid</li>
            <li>Paris</li>
            <li>Rome</li>
          </ul>
        </div>
        <div class="mcq-item">
          <p><strong>Which planet is known as the Red Planet?</strong></p>
          <ul>
            <li>Earth</li>
            <li>Mars</li>
            <li>Jupiter</li>
            <li>Venus</li>
          </ul>
        </div>
      `;
    }, 2000);  // Simulate a 2-second delay for the loader
  });
});











// Move to Top Button
const moveToTopBtn = document.getElementById('moveToTopBtn');

// Show the button when the user scrolls down 200px
window.onscroll = function () {
  if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
    moveToTopBtn.style.display = 'block';
  } else {
    moveToTopBtn.style.display = 'none';
  }
};

// Smooth scroll to top when the button is clicked
moveToTopBtn.addEventListener('click', () => {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
});
