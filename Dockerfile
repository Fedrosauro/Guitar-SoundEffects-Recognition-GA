FROM python:3.11-slim

WORKDIR /app

# Update package lists and install FluidSynth and ffmpeg
RUN apt-get update && \
    apt-get install -y fluidsynth ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt first to leverage Docker's caching for dependencies
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to start a shell
CMD ["/bin/bash"]
