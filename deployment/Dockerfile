FROM public.ecr.aws/lambda/python:3.12

# Install C++ compilers needed to build the SHAP library
RUN dnf install -y gcc gcc-c++

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

# Copy your application code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}

# Copy the model artifacts
COPY model_artifacts_002.pkl ${LAMBDA_TASK_ROOT}

CMD [ "app.lambda_handler" ]