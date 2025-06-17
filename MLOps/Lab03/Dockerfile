FROM public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

COPY cifar10_model.pt lambda.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["lambda.lambda_handler"]
