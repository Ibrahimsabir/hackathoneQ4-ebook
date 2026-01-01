import React, { useState, useRef, useEffect } from 'react';
import Layout from '@theme/Layout';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Button, LoadingDots } from '../components/ui';
import styles from './chatbot.module.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function Chatbot(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI assistant for the **Physical AI & Humanoid Robotics** book. Ask me anything about the content!',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-expand textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage: Message = {
          role: 'assistant',
          content: data.answer || 'I couldn\'t generate a response. Please try again.',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: '⚠️ **Error:** Unable to connect to the backend server. Please make sure it\'s running on `http://localhost:8000`',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add a toast notification here in future
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Hello! I\'m your AI assistant for the **Physical AI & Humanoid Robotics** book. Ask me anything about the content!',
        timestamp: new Date(),
      },
    ]);
  };

  const charCount = input.length;
  const maxChars = 1000;

  return (
    <Layout
      title="AI Chatbot"
      description="Ask questions about Physical AI & Humanoid Robotics">
      <div className={styles.chatbotContainer}>
        {/* Header */}
        <motion.div
          className={styles.chatbotHeader}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className={styles.headerContent}>
            <h1 className={styles.title}>
              <span className={styles.icon}>🤖</span>
              <span className={styles.gradientText}>AI Chatbot Assistant</span>
            </h1>
            <p className={styles.subtitle}>Ask me anything about Physical AI & Humanoid Robotics!</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={clearChat}
            aria-label="Clear conversation"
          >
            Clear Chat
          </Button>
        </motion.div>

        {/* Messages Container */}
        <div className={styles.messagesContainer}>
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={index}
                className={`${styles.message} ${
                  message.role === 'user' ? styles.userMessage : styles.assistantMessage
                }`}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className={styles.messageHeader}>
                  <span className={styles.messageRole}>
                    {message.role === 'user' ? '👤 You' : '🤖 AI Assistant'}
                  </span>
                  <span className={styles.timestamp}>
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                <div className={styles.messageContent}>
                  {message.role === 'assistant' ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={vscDarkPlus}
                              language={match[1]}
                              PreTag="div"
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    <p>{message.content}</p>
                  )}
                </div>
                {message.role === 'assistant' && (
                  <button
                    className={styles.copyButton}
                    onClick={() => copyToClipboard(message.content)}
                    aria-label="Copy message"
                    title="Copy to clipboard"
                  >
                    📋
                  </button>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Loading Indicator */}
          {isLoading && (
            <motion.div
              className={styles.loadingMessage}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className={styles.messageHeader}>
                <span className={styles.messageRole}>🤖 AI Assistant</span>
              </div>
              <div className={styles.loadingContent}>
                <LoadingDots size="md" />
                <span className={styles.loadingText}>Thinking...</span>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Container */}
        <motion.div
          className={styles.inputContainer}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className={styles.inputWrapper}>
            <textarea
              ref={textareaRef}
              className={styles.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your question here... (Press Enter to send, Shift+Enter for new line)"
              rows={1}
              disabled={isLoading}
              maxLength={maxChars}
            />
            <div className={styles.charCounter}>
              <span className={charCount > maxChars * 0.9 ? styles.charWarning : ''}>
                {charCount}/{maxChars}
              </span>
            </div>
          </div>
          <Button
            variant="primary"
            size="md"
            onClick={handleSend}
            disabled={isLoading || !input.trim() || charCount > maxChars}
            isLoading={isLoading}
            aria-label="Send message"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </Button>
        </motion.div>

        {/* Info Section */}
        <motion.div
          className={styles.info}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <p>
            <strong>💡 Tip:</strong> The AI understands markdown formatting and can provide code examples
          </p>
          <p>
            <strong>⚙️ Backend:</strong> http://localhost:8000 | <strong>Model:</strong> Mistral AI (via OpenRouter) | <strong>Vector Store:</strong> Qdrant (467 embeddings)
          </p>
        </motion.div>
      </div>
    </Layout>
  );
}
