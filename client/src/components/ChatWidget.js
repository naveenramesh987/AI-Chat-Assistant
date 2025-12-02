import React, { useState, useEffect, useRef } from "react";
import { FaRobot, FaPaperPlane, FaTimes, FaCommentDots } from "react-icons/fa";

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [threadId, setThreadId] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const initialMessages = [
        {
          text: "Hello! I'm your shopping assistant. How can I help you today?",
          isAgent: true,
        },
      ];
      setMessages(initialMessages);
    }
  }, [isOpen, messages.length]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  console.log(messages);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    console.log(inputValue);

    const message = {
      text: inputValue,
      isAgent: false,
    };

    setMessages((prevMessages) => [...prevMessages, message]);
    setInputValue("");

    const endpoint = threadId
      ? `http://localhost:8000/chat/${threadId}`
      : "http://localhost:8000/chat";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: inputValue,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Success:", data);

      const agentResponse = {
        text: data.response,
        isAgent: true,
        threadId: data.threadId,
      };

      setMessages((prevMessages) => [...prevMessages, agentResponse]);
      setThreadId(data.threadId);
      console.log(messages);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className={`chat-widget-container ${isOpen ? "open" : ""}`}>
      {isOpen ? (
        <>
          <div className="chat-header">
            <div className="chat-title">
              <FaRobot />
              <h3>Shop Assistant</h3>
            </div>
            <button className="close-button" onClick={toggleChat}>
              <FaTimes />
            </button>
          </div>

          <div className="chat-messages">
            {messages.map((message, index) => (
              <div key={index}>
                <div
                  className={`message ${
                    message.isAgent ? "message-bot" : "message-user"
                  }`}
                >
                  {message.text}
                </div>
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>

          <form className="chat-input-container" onSubmit={handleSendMessage}>
            <input
              type="text"
              className="message-input"
              placeholder="Type your message..."
              value={inputValue}
              onChange={handleInputChange}
            />
            <button
              type="submit"
              className="send-button"
              disabled={inputValue.trim() === ""}
            >
              <FaPaperPlane size={16} />
            </button>
          </form>
        </>
      ) : (
        <button className="chat-button" onClick={toggleChat}>
          <FaCommentDots />
        </button>
      )}
    </div>
  );
};

export default ChatWidget;
