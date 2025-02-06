"use client";
import { useState, useEffect, useRef } from "react";
import Header from "./components/Header";
import Message from "./components/Message";
import Input from "./components/Input";
import DarkModeToggle from "./components/DarkModeToggle";

interface ChatMessage {
  id: number;
  content: string;
  role: "user" | "bot";
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  // Handle sending message
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessage: ChatMessage = { id: Date.now(), content: input, role: "user" };
    setMessages((prev) => [...prev, newMessage]);
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/query", {
        method: "POST", // Changed to POST
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }), // Send input in the body
      });

      if (!res.ok) throw new Error("Failed to fetch response");

      const data = await res.json();
      const botMessage: ChatMessage = { id: Date.now() + 1, content: data.answer, role: "bot" };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 2, content: "⚠️ Error fetching response. Please try again.", role: "bot" },
      ]);
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  return (
    <div className={`min-h-screen ${darkMode ? "dark" : ""}`}>
      <div className="max-w-3xl mx-auto p-4 bg-white dark:bg-gray-800 min-h-screen flex flex-col">
        <Header />
        <DarkModeToggle darkMode={darkMode} setDarkMode={setDarkMode} />
        <div className="flex-grow overflow-y-auto mb-4 space-y-4">
          {messages.map((message) => (
            <Message key={message.id.toString()} message={{ ...message, id: message.id.toString() }} />
          ))}
          {loading && <p className="text-gray-500">Thinking...</p>}
          <div ref={messagesEndRef} />
        </div>
        <Input input={input} handleInputChange={handleInputChange} handleSubmit={handleSubmit} />
      </div>
    </div>
  );
}
