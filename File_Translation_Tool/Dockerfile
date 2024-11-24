# File_Translation_Tool/Dockerfile
FROM python:3.9

# Install required system packages
RUN apt-get update && apt-get install -y xvfb

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Run the GUI application
CMD ["python", "file_translation.py"]

# Add Xvfb startup command to the entry point
# CMD ["sh", "-c", "Xvfb :1 -screen 0 1024x768x24 -ac & python file_translation.py"]
