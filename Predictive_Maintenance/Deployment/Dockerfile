# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file and install dependencies
COPY Predictive_Maintenance/Deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY Predictive_Maintenance/Deployment/app.py .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
