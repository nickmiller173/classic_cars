# 1. Start with the AWS base image for Python 3.9
FROM public.ecr.aws/lambda/python:3.9

# 2. Copy the requirements file first (for caching speed)
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 3. Install the libraries
RUN pip install -r requirements.txt

# 4. Copy your actual code and your specific artifact files
COPY app.py utils.py model_artifacts_001.pkl encoding_artifacts_001.pkl ${LAMBDA_TASK_ROOT}/

# 5. Tell the container to run the 'lambda_handler' function inside 'app.py'
CMD [ "app.lambda_handler" ]