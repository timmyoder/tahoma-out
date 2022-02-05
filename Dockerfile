# container for testing heroku deployment
FROM heroku/heroku:20-build
RUN apt-get update && apt-get install -y python3.8 python3-pip
RUN git clone https://github.com/timmyoder/tahoma-out.git
RUN cd tahoma-out && pip install -r requirements.txt
WORKDIR /tahoma-out
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
