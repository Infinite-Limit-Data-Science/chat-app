FROM python:3.12.3

ENV CHAT_API_VERSION 1.0.0

# docker container run --rm -it python:3.12 bash
# root@ecd39a9762de:/# pwd
# /
WORKDIR /chat-app/api

COPY ./api/Pipfile ./api/Pipfile.lock ./

RUN pip install --no-cache-dir pipenv

RUN pipenv install --system --deploy

COPY --chown=1001:0 ./api ./

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

USER 1001