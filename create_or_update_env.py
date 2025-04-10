import json
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

with open('config.json', 'r') as f:
    config = json.load(f)

subscription_id = config["subscription_id"]
resource_group = config["resource_group"]
workspace_name = config["workspace_name"]

# Authenticate and create the ML client
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Create an environment object using the conda file
env = Environment(
    name="my-ml-env",
    conda_file="./environment.yml",
    image="mcr.microsoft.com/azureml/base:latest",
    description="A sample environment with Python, scikit-learn, pandas, and numpy"
)

# Register (or update) the environment in your workspace
registered_env = ml_client.environments.create_or_update(env)
print(f"Registered environment: {registered_env.name}, version: {registered_env.version}")
