from pydantic import BaseModel, Field
from instructor.exceptions import InstructorRetryException
import instructor
import pytest
import json
from unittest.mock import Mock

class UserProfile(BaseModel):
    name: str = Field(..., description="The user's name")
    age: int = Field(..., description="The user's age")
    bio: str = Field(..., description="The user's biography")

def test_retry_on_validation(mock_bedrock_client):
    """Test retry behavior on validation failure."""
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

    # First response will be a mock error response
    mock_error_response = {
        'body': json.dumps({
            'results': [{
                'outputText': json.dumps({
                    'error': 'Invalid data format',
                    'details': 'Missing required fields'
                })
            }]
        }).encode('utf-8')
    }

    def side_effect(*args, **kwargs):
        if not hasattr(side_effect, 'called'):
            side_effect.called = True
            return mock_error_response
        return mock_response

    mock_bedrock_client.invoke_model.side_effect = side_effect
    instructor_client = instructor.from_boto3(client=mock_bedrock_client)

    try:
        response = instructor_client.chat.completions.create(
            model='anthropic.claude-v2',
            messages=[{"role": "user", "content": "test prompt"}],
            max_retries=1,
            response_model=UserProfile
        )
    except InstructorRetryException as e:
        assert "Invalid data format" in str(e)
        assert "Missing required fields" in str(e)