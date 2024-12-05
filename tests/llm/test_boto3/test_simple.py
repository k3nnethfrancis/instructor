from typing import Dict, Any, Literal
import json
import pytest
from unittest.mock import Mock, patch
import instructor
from instructor import Mode
from pydantic import BaseModel, Field, field_validator
from instructor.function_calls import OpenAISchema
from instructor.exceptions import InstructorRetryException

class UserProfile(OpenAISchema):
    name: str
    age: int
    bio: str

    @field_validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v

def test_from_boto3_initialization(mock_bedrock_client):
    """Test basic client initialization."""
    client = mock_bedrock_client
    instructor_client = instructor.from_boto3(client=client)
    assert instructor_client.client is client

def test_basic_completion(mock_bedrock_client):
    """Test basic completion with JSON response."""
    # Mock response from invoke_model
    mock_response = {
        'body': json.dumps({
            'results': [{
                'outputText': json.dumps({
                    'name': 'John Doe',
                    'age': 30,
                    'bio': 'A software engineer'
                })
            }]
        }).encode('utf-8')
    }

    # Set up the mock response
    mock_bedrock_client.invoke_model.return_value = mock_response

    # Initialize the instructor client
    instructor_client = instructor.from_boto3(client=mock_bedrock_client)

    # Perform the request
    response = instructor_client.chat.completions.create(
        model='anthropic.claude-v2',
        messages=[
            {"role": "user", "content": "test prompt"}
        ],
        response_model=UserProfile
    )

    # Assertions
    assert response.name == 'John Doe'
    assert response.age == 30
    assert response.bio == 'A software engineer'

    # Verify the call
    mock_bedrock_client.invoke_model.assert_called_once()

def test_basic_completion_with_response(mock_bedrock_client):
    """Test completion with raw response access."""
    client = mock_bedrock_client

    mock_response = {
        'body': json.dumps({
            'results': [{
                'outputText': json.dumps({
                    'name': 'John Doe',
                    'age': 30,
                    'bio': 'A software engineer'
                })
            }]
        }).encode('utf-8')
    }

    # Set up the mock response
    client.invoke_model.return_value = mock_response

    # Initialize the instructor client
    instructor_client = instructor.from_boto3(client=client)

    # Perform the request
    response = instructor_client.chat.completions.create(
        model='anthropic.claude-v2',
        messages=[
            {"role": "user", "content": "test prompt"}
        ],
        response_model=UserProfile
    )

    # Assertions
    assert response.name == 'John Doe'
    assert response.age == 30
    assert response.bio == 'A software engineer'

    # Verify the call
    client.invoke_model.assert_called_once()

def test_literal_response(mock_bedrock_client):
    """Test Literal type responses."""
    client = mock_bedrock_client

    mock_response = {
        'contentType': 'application/json',
        'body': json.dumps({
            'results': [{
                'outputText': json.dumps({
                    'content': 'option1'
                })
            }]
        }).encode('utf-8')
    }

    client.invoke_model.return_value = mock_response
    instructor_client = instructor.from_boto3(client=client)

    response = instructor_client.chat.completions.create(
        model="anthropic.claude-v2",
        response_model=Literal["option1", "option2", "option3"],
        messages=[{"role": "user", "content": "Choose an option"}]
    )

    assert response == "option1"