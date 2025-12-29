# Physical AI & Humanoid Robotics - Comprehensive Guide

This repository contains a comprehensive ebook on Physical AI & Humanoid Robotics with an integrated RAG (Retrieval-Augmented Generation) chatbot system, built using a modern static website generator.

## About This Project

This ebook provides a deep dive into the integration of artificial intelligence with physical robotic systems, focusing on humanoid platforms and embodied AI applications. The content is structured into five comprehensive modules:

- **Module 1**: ROS 2 – The Robotic Nervous System
- **Module 2**: Digital Twins – Gazebo & Unity
- **Module 3**: AI Robot Brain – NVIDIA Isaac
- **Module 4**: Vision–Language–Action (VLA)
- **Module 5**: Autonomous Humanoid System

## Book Assistant RAG Chatbot

This project includes an advanced RAG chatbot system that allows users to interact with the book content through natural language queries. The system provides:

- **Agentic RAG System**: Conversational AI with memory and tools for retrieval, summarization, and citation
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Text Selection Queries**: Ask questions specifically about selected text
- **Citation Generation**: Provides references to relevant book sections
- **Session Analytics**: Track conversation history and performance metrics

### Features

- **Interactive Q&A**: Ask questions about the book content and receive accurate, cited responses
- **Context-Aware**: Maintains conversation history for multi-turn interactions
- **Text Selection**: Ask questions about specific text passages you've selected
- **Citations**: All responses include references to relevant book sections
- **Secure & Scalable**: Rate limiting and input sanitization for security

### How to Use

1. Navigate to the "Book Assistant" link in the navigation bar
2. Type your questions about the book content in the chat interface
3. Select text in any book page and ask specific questions about that content
4. View responses with citations to relevant book sections

## Installation

```bash
# Install frontend dependencies
npm install
# or
yarn install
```

For the backend RAG system, install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

## Local Development

### Frontend
```bash
npm run start
# or
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Backend
To run the RAG chatbot backend:
```bash
cd backend
python main.py
```
The backend API will be available at `http://localhost:8000`

## Building the Site

```bash
npm run build
# or
yarn build
```

This command generates static content into the `build` directory and can be served using any static hosting service.

## Backend Configuration

The RAG chatbot system requires the following environment variables in `backend/.env`:

- `COHERE_API_KEY`: Your Cohere API key for embeddings and generation
- `QDRANT_URL`: Your Qdrant vector database URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `NEON_DB_URL`: Your Neon Postgres connection string

## API Endpoints

- `POST /chat`: Main chat endpoint for general book content queries
- `POST /selection-chat`: Endpoint for queries about selected text
- `GET /analytics/session-summary`: Get analytics for a specific session

## Features

- Modern, clean design with excellent readability
- Responsive layout for all device sizes
- Fast loading times
- SEO optimized
- Accessible and inclusive design
- Comprehensive documentation structure

## Vercel Deployment

This project is optimized for deployment on Vercel. The following environment is configured:

- Framework: Docusaurus
- Build Command: `npm run build` or `yarn build`
- Output Directory: `build`
- Install Command: `npm install` or `yarn install`

## Contributing

If you'd like to contribute to this ebook, please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
