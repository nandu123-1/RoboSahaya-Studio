FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROBOSAHAYA_MODE=mock
ENV ROS_DOMAIN_ID=0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml requirements.txt README.md /app/
COPY src/ /app/src/
COPY tests/ /app/tests/

# Install python requirements
RUN pip3 install --break-system-packages -r requirements.txt

EXPOSE 7860

CMD ["python3", "src/robosahaya/app.py", "--host", "0.0.0.0", "--port", "7860"]
