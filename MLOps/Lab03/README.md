# MLOps – CIFAR-10 Lambda Inference on AWS

This repository contains a complete setup for deploying a **CIFAR-10 image classification model** using **AWS Lambda with Docker**, integrated with **S3** and **ECR**.

> ⚠️ NOTE: You can skip **Steps 1–5** and **Step 8** by using the provided scripts:
>
> - `python3 helpers/uploadToECR.py`
> - `python3 helpers/createLambda.py`

---

## Prerequisites

1. **AWS CLI** – Install and configure using the [official AWS CLI setup guide](https://docs.aws.amazon.com/cli/v1/userguide/install-linux.html).
2. Ensure your AWS credentials and config are set under `~/.aws/`.

---

## 1. Build Docker Image

```bash
docker build -t cifar10-lambda .
```

## 2. Create a Remote Repository (ECR)

```bash
aws ecr create-repository --repository-name cifar10-lambda
```

## 3. Authenticate Docker with ECR

```bash
aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com
```

## 4. Tag the Docker Image

```bash
docker tag cifar10-lambda:latest {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/cifar10-lambda:latest
```

## 5. Push the Image to ECR

```bash
docker push {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/cifar10-lambda:latest
```

## 6. Verify Uploaded Image

```bash
aws ecr describe-images --repository-name cifar10-lambda
```

## 7. Test Lambda Image Locally

### 7.1. Create Lambda IAM Role

```bash
aws iam create-role --role-name Basic-Lambda-Role --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'
```

### 7.2. Attach Execution Policy

```bash
aws iam attach-role-policy \
  --role-name Basic-Lambda-Role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### 7.3. Allow S3 Access

```bash
aws iam attach-role-policy \
  --role-name Basic-Lambda-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 7.4. Create S3 Bucket

```bash
aws s3 mb s3://{BUCKET_NAME}
```

### 7.5. Upload an Image

```bash
aws s3 cp 4915.png s3://{BUCKET_NAME}/{FILE_NAME}
```

### 7.6. Run Docker Locally

```bash
docker run -p 9000:8080 -v ~/.aws:/root/.aws:ro cifar10-lambda:latest
```

### 7.7. Invoke the Lambda Locally

```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
-H "Content-Type: application/json" \
-d '{
  "Records": [
    {
      "s3": {
        "bucket": {
          "name": "{BUCKET_NAME}"
        },
        "object": {
          "key": "{FILE_NAME}"
        }
      }
    }
  ]
}'
```

## 8. Deploy to AWS Lambda

```bash
aws lambda create-function \
  --function-name cifar10-image-inference \
  --package-type Image \
  --code ImageUri={ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/cifar10-lambda:latest \
  --role arn:aws:iam::{ACCOUNT_ID}:role/Basic-Lambda-Role \
  --timeout 30 \
  --memory-size 1024
```

### 8.1 List Functions

```bash
aws lambda list-functions
```

### 8.2 Invoke AWS Lambda

```bash
aws lambda invoke \
  --function-name cifar10-image-inference \
  --cli-binary-format raw-in-base64-out \
  --payload '{"Records":[{"s3":{"bucket":{"name":"{BUCKET_NAME}"},"object":{"key":"{FILE_NAME}"}}}]}' \
  response.json
```

### 8.3 Check Output

```bash
aws s3 ls s3://{BUCKET_NAME}/
aws s3 cp s3://{BUCKET_NAME}/{FILE_NAME}_result.json .
cat {FILE_NAME}_result.json
```

Or use the AWS Console:

- Visit https://aws.amazon.com/console/
- Go to S3 → {BUCKET_NAME} → View the files.

## 9. Automate Lambda Trigger on New Upload

### 9.1 Grant Permission to S3

```bash
aws lambda add-permission \
  --function-name cifar10-image-inference \
  --principal s3.amazonaws.com \
  --statement-id 1 \
  --action "lambda:InvokeFunction" \
  --source-arn arn:aws:s3:::{BUCKET_NAME} \
  --source-account {ACCOUNT_ID}
```

### 9.2 Add Notification Configuration

```bash
aws s3api put-bucket-notification-configuration \
  --bucket {BUCKET_NAME} \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [
      {
        "LambdaFunctionArn": "arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:cifar10-image-inference",
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
          "Key": {
            "FilterRules": [
              {
                "Name": "suffix",
                "Value": ".png"
              }
            ]
          }
        }
      }
    ]
  }'
```

## 10. Final Test – Upload a New Image

```bash
aws s3 cp {LOCAL_IMAGE_FILE} s3://{BUCKET_NAME}/
```

This should automatically invoke your Lambda function and produce an output .json file.
