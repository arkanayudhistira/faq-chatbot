# Use Python 3.12 image for Docker
FROM python:3.12-slim

# Set project directory
WORKDIR /faq-chatbot

# Copy all files to docker
COPY . .

# Install required python libraries from requirements.txt file
RUN pip install -r requirements.txt

# Set docker to listen to streamlit default port (8501)
EXPOSE 8501

# Set docker to check if the streamlit is healthy or unhealthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the streamlit run command for the app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]