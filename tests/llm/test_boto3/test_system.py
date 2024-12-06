import pytest
import instructor
from pydantic import BaseModel, Field
import json


class User(BaseModel):
    name: str
    age: int


def test_from_boto3_initialization(mock_bedrock_client):
    """Test that the client is properly patched with Instructor capabilities."""
    instructor_client = instructor.from_boto3(client=mock_bedrock_client)
    assert hasattr(instructor_client.chat.completions, 'create')


def test_client_from_boto3_with_response(mock_bedrock_client):
    """Test basic completion with response object."""
    client = instructor.from_boto3(
        mock_bedrock_client,
        max_tokens=1000,
        model="anthropic.claude-v2",
    )
    
    mock_bedrock_client.invoke_model.return_value = {
        'body': json.dumps({
            'id': 'msg_123',
            'type': 'message',
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': json.dumps({
                    'name': 'Jason',
                    'age': 10
                })
            }]
        }).encode()
    }

    user, response = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    
    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, dict)


def test_client_boto3_response(mock_bedrock_client):
    """Test basic response without completion object."""
    client = instructor.from_boto3(
        mock_bedrock_client,
        max_tokens=1000,
        model="anthropic.claude-v2",
    )
    
    mock_bedrock_client.invoke_model.return_value = {
        'body': json.dumps({
            'content': [{
                'text': json.dumps({
                    'name': 'Jason',
                    'age': 10
                })
            }]
        }).encode()
    }

    user = client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    
    assert user.name == "Jason"
    assert user.age == 10