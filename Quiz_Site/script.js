// script.js

let urls = [];

function addUrlInput() {
  const urlInputs = document.getElementById('urlInputs');
  const input = document.createElement('input');
  input.type = 'url';
  input.name = 'url';
  urlInputs.appendChild(input);
}

function removeUrlInput() {
  const urlInputs = document.getElementById('urlInputs');
  if (urlInputs.children.length > 0) {
    urlInputs.removeChild(urlInputs.lastChild);
  }
}

document.getElementById('urlForm').addEventListener('submit', function(event) {
  event.preventDefault();
  const formData = new FormData(this);
  const urlValues = formData.getAll('url');
  urls = urlValues;
  console.log(urls);
  sendUrlsToApi(urls);
});

function sendUrlsToApi(urls) {
    fetch('http://192.168.1.13:5000/quiz-generator', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ urls: urls }),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Response from API:', data);
        displayResponse(data);
      })
      .catch(error => {
        console.error('Error sending URLs to API:', error);
      });
  }
  
  function displayResponse(responseObject) {
    const responseContainer = document.getElementById('responseContainer');
    responseContainer.innerHTML = ''; // Clear previous content
  
    if (Array.isArray(responseObject.mcqs)) {
      responseObject.mcqs.forEach(element => {
        const textarea = document.createElement('textarea');
        textarea.textContent = element;
        responseContainer.appendChild(textarea);
  
        const addButton = document.createElement('button');
        addButton.textContent = 'Add MCQ';
        addButton.className = 'add-mcq-button';
        addButton.onclick = function() {
          sendMcqToApi(textarea.value);
        };
        responseContainer.appendChild(addButton);
      });
    } else {
      console.error('Invalid response format:', responseObject);
    }
  }
  
  function sendMcqToApi(mcqContent) {
    fetch('http://192.168.1.13:5000/submit', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ mcq: mcqContent }),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Response from API:', data);
        // Handle response as needed
      })
      .catch(error => {
        console.error('Error sending MCQ to API:', error);
      });
  }
  