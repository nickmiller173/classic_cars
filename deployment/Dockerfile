FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

# Copy your application code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}

# Copy the model artifacts
COPY model_artifacts_002.pkl ${LAMBDA_TASK_ROOT}
COPY encoding_artifacts_002.pkl ${LAMBDA_TASK_ROOT}

CMD [ "app.lambda_handler" ]