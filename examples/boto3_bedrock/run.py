from pydantic import BaseModel
import boto3
import instructor
from instructor.function_calls import openai_schema

# Initialize the Bedrock client
bedrock = boto3.client('bedrock-runtime')

# Patch the client with instructor
client = instructor.from_boto3(bedrock)


@openai_schema
class Properties(BaseModel):
    key: str
    value: str


@openai_schema
class User(BaseModel):
    name: str
    age: int
    properties: list[Properties]


user = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_tokens=1024,
    max_retries=2,
    messages=[
        {
            "role": "user",
            "content": "Create a user named John who is 30 years old. with properties: email=john@example.com and role=user",
        }
    ],
    response_model=User,
)

print(user.model_dump_json(indent=2))