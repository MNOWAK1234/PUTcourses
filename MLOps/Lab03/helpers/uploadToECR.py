import boto3
import docker
import subprocess

REGION = "eu-west-1"

ecr_client = boto3.client("ecr")
docker_client = docker.from_env()
aws_account_id = boto3.client("sts").get_caller_identity().get("Account")


def create_or_get_ecr_repository(repository_name):
    try:
        response = ecr_client.create_repository(repositoryName=repository_name)
        print(f"Repository {repository_name} created.")
        repository_uri = response["repository"]["repositoryUri"]
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository {repository_name} already exists.")
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repository_uri = response["repositories"][0]["repositoryUri"]
    return repository_uri


# Authenticate Docker to ECR
def authenticate_ecr(region, aws_account_id):
    token = ecr_client.get_authorization_token()["authorizationData"][0]
    ecr_url = token["proxyEndpoint"]
    print(f"Authenticating Docker to {ecr_url}...")
    login_command = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_url}"
    subprocess.run(login_command, shell=True, check=True)
    print("Docker authenticated with ECR")


def build_and_push_docker_image(repository_uri, repository_name, image_tag, dockerfile_path="../"):
    print(f"Building Docker image {repository_name}:{image_tag}...")
    docker_client.images.build(path=dockerfile_path, tag=f"{repository_name}:{image_tag}")
    print("Docker image built successfully.")

    # Tag the image to match the repository URI
    full_image_name = f"{repository_uri}:{image_tag}"
    docker_client.images.get(f"{repository_name}:{image_tag}").tag(full_image_name)
    print(f"Tagged image: {full_image_name}")

    print(f"Pushing image to ECR repository {repository_uri}...")
    for line in docker_client.images.push(
        repository_uri, tag=image_tag, stream=True, decode=True
    ):
        print(line)


if __name__ == "__main__":
    repository_name = "cifar10-lambda"  # The name of your repository
    image_tag = "latest"  # The image tag (e.g., latest)
    dockerfile_path = "../"  # Points to the parent directory

    # Create or get the ECR repository URI
    repository_uri = create_or_get_ecr_repository(repository_name)
    print(f"Repository URI: {repository_uri}")

    # Authenticate Docker with ECR
    authenticate_ecr(REGION, aws_account_id)

    # Build the Docker image and push it to ECR
    build_and_push_docker_image(repository_uri, repository_name, image_tag, dockerfile_path)
