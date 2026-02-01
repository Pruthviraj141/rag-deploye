const chat = document.getElementById("chat");
const statusEl = document.getElementById("status");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const chips = document.querySelectorAll(".chip");

function setStatus(text, type = "idle") {
  statusEl.textContent = text;
  statusEl.style.color = type === "error" ? "#ff6b6b" : "#9fb0c3";
}

function addMessage(text, sender = "bot") {
  const message = document.createElement("div");
  message.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  message.appendChild(bubble);
  chat.appendChild(message);
  chat.scrollTop = chat.scrollHeight;
}

async function ask(question) {
  if (!question.trim()) return;

  addMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  setStatus("Thinking...");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    const data = await res.json();
    addMessage(data.answer || "Information not available", "bot");
    setStatus("Ready");
  } catch (err) {
    addMessage("Sorry, something went wrong.", "bot");
    setStatus("Error", "error");
  } finally {
    sendBtn.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(input.value);
});

chips.forEach((chip) => {
  chip.addEventListener("click", () => ask(chip.textContent));
});
