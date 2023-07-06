FROM python:3.8
USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/
RUN pip3 install -r requirements.txt
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN airflow db init
RUN airflow users create -e harshalkumre1998@gmail.com -f harshal -l kumre -p admin -r Admin -u admin
RUN chmod 777 start.sh
RUN apt-get update -y && apt install awscli -y 
ENTRYPOINT ["/bin/sh"]
EXPOSE 5000
CMD ["start.sh", "python3", "app.py"]