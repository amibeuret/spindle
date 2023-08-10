# Use an official Python runtime as a parent image
FROM python:3.7.3

# Set the working directory to /app
WORKDIR /DeepSleep

# Copy the current directory contents into the container at /app
COPY deepsleep /DeepSleep

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt
run pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 5000

RUN ls

RUN pwd

# Run app.py when the container launches
CMD ["python", "run.py", "start_service", "deepsleep/configs/config.yaml"]
