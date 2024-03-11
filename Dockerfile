FROM python:3.8
WORKDIR /my_project
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY script.py /my_project/script.py
COPY train_df.csv /my_project/train_df.csv
COPY test_df.csv /my_project/test_df.csv
CMD ["python", "my_project/script.py"]