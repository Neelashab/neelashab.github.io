import React, { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

export default function WorkoutAssistant() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [assistantId, setAssistantId] = useState(null);
  const [threadId, setThreadId] = useState(null);

  const API_BASE = "https://workout-assistant-cdua.onrender.com";

  // On first load, create assistant + thread
  useEffect(() => {
    const initAssistantAndThread = async () => {
      try {
        const assistantRes = await axios.post(`${API_BASE}/assistants`, {
          graph_id: "memory-agent",
          config: {
            configurable: {
              user_id: "demo_user",
            },
          },
          name: "Workout Assistant",
          if_exists: "do_nothing",
        });

        const createdAssistantId = assistantRes.data.assistant_id;
        setAssistantId(createdAssistantId);

        const threadRes = await axios.post(`${API_BASE}/threads`, {
          metadata: {
            user: "demo_user",
          },
        });

        const createdThreadId = threadRes.data.thread_id;
        setThreadId(createdThreadId);
      } catch (error) {
        console.error("Error initializing assistant and thread:", error);
        setResponse("Failed to initialize assistant.");
      }
    };

    initAssistantAndThread();
  }, []);

  const handleSend = async () => {
    if (!message.trim() || !assistantId || !threadId) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await axios.post(
        `${API_BASE}/threads/${threadId}/runs/wait`,
        {
          assistant_id: assistantId,
          input: {
            messages: [
              {
                role: "user",
                content: message,
              },
            ],
          },
          config: {
            configurable: {
              user_id: "demo_user",
            },
          },
        }
      );

      const messages = res.data.messages || [];
      const latestAIResponse = [...messages].reverse().find(m => m.type === "ai");
      setResponse(latestAIResponse?.content || "No response from assistant.");

    } catch (error) {
      console.error("Error sending message:", error);
      setResponse("Failed to get response from assistant.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="assistant-box">
        <h1 className="title">workout_AssIstant</h1>
        <textarea
        className="textarea"
        placeholder="How are you feeling today?"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
    }
  }}
/>

        {response && (
          <pre className="response-box">{response}</pre>
        )}
      </div>
    </div>
  );
}
