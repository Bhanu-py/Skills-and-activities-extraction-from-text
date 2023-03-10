FROM python:3.7-slim
RUN pip install -r requirements.txt

EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
