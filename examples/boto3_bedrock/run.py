import boto3
import instructor
from pydantic import BaseModel
import json

# Define our Pydantic models
class Properties(BaseModel):
    key: str
    value: str

class User(BaseModel):
    name: str
    age: int
    properties: list[Properties]

# Initialize the Bedrock client and patch it with instructor
bedrock = boto3.client('bedrock-runtime')
client = instructor.from_boto3(bedrock, mode=instructor.Mode.BOTO3_TOOLS)

# Create a structured response
response = client.invoke_model(
    modelId="anthropic.claude-v2",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "prompt": "\n\nHuman: Create a user for a model with a name, age, and properties.\n\nAssistant:",
        "max_tokens_to_sample": 1000,
        "temperature": 0.7,
        "anthropic_version": "bedrock-2023-05-31"
    })
)

# Parse and print the response
response_body = json.loads(response['body'].read())
print(json.dumps(response_body, indent=2))