from ml_accelerator.config.env import Env
from ml_accelerator.config.params import Params

def get_image_uri(
    docker_repository_type: str = Env.get("DOCKER_REPOSITORY_TYPE"),
    docker_repository_name: str = Env.get("DOCKER_REPOSITORY_NAME"),
    dockerhub_username: str = Env.get("DOCKERHUB_USERNAME"),
    ecr_repository_uri: str = Env.get("ECR_REPOSITORY_URI"),
    env: str = Env.get("ENV"),
    version: str = Params.VERSION
) -> str:
    if docker_repository_type == "dockerhub":
        return f"{dockerhub_username}/{docker_repository_name}:{env}-image-{version}"
    elif docker_repository_type == "ECR":
        return f"{ecr_repository_uri}/{docker_repository_name}:{env}-image-{version}"
    else:
        raise ValueError(f"Invalid docker_repository_type: {docker_repository_type}")
