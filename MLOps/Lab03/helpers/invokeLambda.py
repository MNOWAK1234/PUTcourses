import boto3
import json
import requests

REGION = "eu-west-1"  # Correct region for your Lambda

client = boto3.client('lambda', region_name=REGION)

def invoke_lambda_using_boto3(function_name: str, payload: dict):
    response = client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload).encode('utf-8')
    )

    response_payload = json.loads(response['Payload'].read())
    print('Response from Lambda:', response_payload)

# Optional: Only use if you're invoking a Lambda URL (not needed with S3 trigger)
def invoke_lambda(lambda_url: str, payload: dict, headers=None):
    if headers is None:
        headers = {
            "Content-Type": "application/json"
        }

    try:
        response = requests.post(lambda_url, json=payload, headers=headers)
        if response.status_code == 200:
            print(response.text)
        else:
            print(f"Error invoking Lambda: {response.status_code} - {response.text}")

    except requests.RequestException as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    payload = {
        "key": "example"
    }

    function_name = 'cifar10-image-inference'
    invoke_lambda_using_boto3(function_name, payload)

    # Lambda URL method not needed for S3 trigger, so this part is optional
    # lambda_url = "https://qcs6h3gsigqg6bslp4zwd7a4640rgkht.lambda-url.eu-west-2.on.aws/"
    # invoke_lambda(lambda_url, payload)
