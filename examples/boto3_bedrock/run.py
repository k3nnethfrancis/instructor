import json
import boto3
import instructor
from pydantic import BaseModel, Field

@instructor.openai_schema
class UserProfile(BaseModel):
    """User profile information."""
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")
    email: str = Field(description="User's email address")
    role: str = Field(description="User's role in the system")

def main():
    bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
    client = instructor.from_boto3(bedrock)
    
    result = client.chat.completions.create(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=1024,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": "Create a user named Ken who is 30 years old with properties: email=ken@agency42.co and role=alchemist"
            }
        ],
        response_model=UserProfile,
    )
    
    print(f"\nType: {type(result)}\n")
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()