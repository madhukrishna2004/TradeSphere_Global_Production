let userId = localStorage.getItem("chat_user_id") || null;

// Send message when button is clicked
document.getElementById("chatbot-send").addEventListener("click", function () {
    const userMessage = document.getElementById("chatbot-input").value.trim();
    if (userMessage === "") return;

    // Display user's message
    displayMessage(userMessage, "user");

    // Clear input field
    document.getElementById("chatbot-input").value = "";

    // Show typing indicator
    document.getElementById("loading-indicator").style.display = 'inline-block';

    // Prepare the payload
    const payload = { message: userMessage };
    if (userId) payload.user_id = userId;

    // Send the message to the backend
    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        // Hide typing indicator
        document.getElementById("loading-indicator").style.display = 'none';

        // Set session user ID if new
        if (data.user_id && !userId) {
            userId = data.user_id;
            localStorage.setItem("chat_user_id", userId);
        }

        // Display the response
        if (data.response) {
            displayMessage(data.response, "bot");
        } else if (data.error) {
            displayMessage("⚠️ " + data.error, "bot");
        }
    })
    .catch(error => {
        document.getElementById("loading-indicator").style.display = 'none';
        console.error("Error:", error);
        displayMessage("❌ Oops! Something went wrong. Please try again.", "bot");
    });
});

// Function to display chat messages
function displayMessage(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chatbot-message", sender === "bot" ? "bot-message" : "user-message");

    // Format Markdown-style bold and line breaks
    const formattedMessage = message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br/>');

    messageDiv.innerHTML = formattedMessage;
    document.getElementById("chatbot-body").appendChild(messageDiv);

    scrollToBottom();
    playMessageSound();
}

// Scroll to the latest message
function scrollToBottom() {
    const chatbotBody = document.getElementById("chatbot-body");
    chatbotBody.scrollTop = chatbotBody.scrollHeight;
}

// Play message notification sound
function playMessageSound() {
    const sound = document.getElementById("message-sound");
    if (sound) {
        sound.play().catch(err => console.log("Audio blocked:", err));
    }
}

// Submit on Enter (except Shift+Enter)
document.getElementById("chatbot-input").addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        document.getElementById("chatbot-send").click();
    }
});

// Scroll on focus
document.getElementById("chatbot-input").addEventListener("focus", scrollToBottom);

// Optional: Reset Chat Session
function resetChatSession() {
    localStorage.removeItem("chat_user_id");
    userId = null;
    displayMessage("🔄 Chat session reset. Start again with a new query.", "bot");
}
