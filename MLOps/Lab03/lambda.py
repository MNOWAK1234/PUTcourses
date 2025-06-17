import json
import torch
from PIL import Image
import torchvision.transforms as transforms
import boto3
import os

# Load model only once when container starts
model = torch.jit.load("cifar10_model.pt")
model.eval()

s3 = boto3.client("s3")

class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def lambda_handler(event, context):
    # Extract bucket and key from event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key    = event['Records'][0]['s3']['object']['key']

    # Download image from S3
    download_path = f"/tmp/{os.path.basename(key)}"
    s3.download_file(bucket, key, download_path)

    # Load and transform image
    image = Image.open(download_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = class_labels[predicted_idx]

    # Save result back to S3
    result_key = f"{key}_result.json"
    result = {
        "predicted_index": predicted_idx,
        "predicted_label": predicted_label
    }
    result_path = f"/tmp/{os.path.basename(result_key)}"
    with open(result_path, "w") as f:
        json.dump(result, f)

    s3.upload_file(result_path, bucket, result_key)

    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
