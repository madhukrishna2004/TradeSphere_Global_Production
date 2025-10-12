function playMessageSound() {
    let sound = document.getElementById("message-sound");
    sound.play().catch(error => console.log("Audio play blocked:", error));
}

function addChatbotMessage(text) {
    let chatBox = document.getElementById("chatbot-body");
    let botMessage = `<div class="chatbot-message bot-message"><strong></strong> ${text}</div>`;
    chatBox.innerHTML += botMessage;
    chatBox.scrollTop = chatBox.scrollHeight;

    // Play sound when a new bot message is added
    playMessageSound();
}

function sendMessage() {
    let userInput = document.getElementById("chatbot-input").value;
    let chatBox = document.getElementById("chatbot-body");

    if (userInput.trim() === "") return;

    let userMessage = `<div class="chatbot-message user-message"><strong></strong> ${userInput}</div>`;
    chatBox.innerHTML += userMessage;
    document.getElementById("chatbot-input").value = "";

    // Check if the response is already in local storage
    let cachedResponse = localStorage.getItem(userInput);
    if (cachedResponse) {
        addChatbotMessage(cachedResponse);
        return;
    }

    // Show loading message
    let loadingMessage = `<div class="chatbot-message bot-message" id="loading">...</div>`;
    chatBox.innerHTML += loadingMessage;
    chatBox.scrollTop = chatBox.scrollHeight;

    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loading").remove(); // Remove loading message
        addChatbotMessage(data.response);

        // Store the response in local storage
        localStorage.setItem(userInput, data.response);
    })
    .catch(error => {
        document.getElementById("loading").remove(); // Remove loading message
        console.error("Error:", error);
    });
}


//
document.addEventListener("DOMContentLoaded", function () {
    const onboardingPopup = document.getElementById("onboarding-popup");
    const popupContent = document.getElementById("popup-content"); // Add this inside the popup div
    const steps = document.querySelectorAll(".step");
    const nextButtons = document.querySelectorAll(".next-step");
    const prevButtons = document.querySelectorAll(".prev-step");
    const finishButton = document.getElementById("finish-onboarding");

    // Open onboarding automatically for new users
    setTimeout(() => {
        onboardingPopup.style.display = "block";
        document.getElementById("step-1").classList.add("active-step");
    }, 1000);

    // Navigation
    nextButtons.forEach((button) => {
        button.addEventListener("click", function () {
            const nextStep = document.getElementById(this.dataset.next);
            steps.forEach(step => step.classList.remove("active-step"));
            nextStep.classList.add("active-step");
        });
    });

    prevButtons.forEach((button) => {
        button.addEventListener("click", function () {
            const prevStep = document.getElementById(this.dataset.prev);
            steps.forEach(step => step.classList.remove("active-step"));
            prevStep.classList.add("active-step");
        });
    });

    // Finish onboarding
    finishButton.addEventListener("click", () => {
        onboardingPopup.style.display = "none";
    });

    // Close popup when clicking outside (EXACT FIX HERE)
    window.addEventListener("click", function (event) {
        if (event.target === onboardingPopup) {
            onboardingPopup.style.display = "none";
        }
    });

    // Prevent closing when clicking inside the popup
    popupContent.addEventListener("click", function (event) {
        event.stopPropagation();
    });
});
