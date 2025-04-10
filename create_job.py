import json
from azure.ai.ml import MLClient, Input, command
from azure.identity import DefaultAzureCredential

with open('config.json', 'r') as f:
    config = json.load(f)

subscription_id = config["subscription_id"]
resource_group_name = config["resource_group_name"]
workspace_name = config["workspace_name"]
compute = config["compute"]

# Authenticate and create a client
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id=subscription_id,
                     resource_group_name=resource_group_name,
                     workspace_name=workspace_name)

# Define the job
job = command(
    code="../quantifier_recommender",  # Directory containing your source code
    command="python __run_experiments.py",
    environment="quantifier-recommender-env:9",
    compute=compute
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted, run_id:", returned_job.id)
