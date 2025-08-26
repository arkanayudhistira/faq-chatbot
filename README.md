# FAQ Chatbot using RAG Agentic LLM

An interactive FAQ chatbot that answers user questions from internal FAQ documents.

## How to Use

1. Clone the repository and navigate to the project director
   
  ```
  git clone https://github.com/arkanayudhistira/faq-chatbot.git
  cd faq-chatbot
  ```

2. Open the .env  file and replace the [GOOGLE_API_KEY](https://aistudio.google.com/app/apikey) and [PINECONE_API_KEY](https://app.pinecone.io/) with the API key from your account
3. Run this command to build the Docker Image from the Dockerfile

  ```
  docker build -t faq-chatbot .
  ```

4. Run the Docker container using this command:

  ```
  docker run -p 8501:8501 faq-chatbot
  ```

5. Open the chatbot app by browsing to http://localhost:8501

