from typing import Annotated
from pydantic import AfterValidator, BaseModel, Field
import pytest
import instructor
from itertools import product
import json
from instructor import Mode
from unittest.mock import AsyncMock
from pydantic import ValidationError

# Test configuration
models = ["anthropic.claude-v2"]
modes = [Mode.BOTO3_JSON, Mode.BOTO3_TOOLS]


def uppercase_validator(v: str):
    """Validate that a string is uppercase."""
    if not v.isupper():
        raise ValueError(
            "All letters in the name should be in uppercase (e.g., TOM, JONES) instead of tom, jones"
        )
    return v.strip()


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_upper_case(model, mode, mock_bedrock_client):
    """Test basic retry functionality with uppercase validation."""
    client = instructor.from_boto3(mock_bedrock_client, mode=mode)
    
    # First response with lowercase name (should trigger retry)
    mock_error_response = {
        'body': json.dumps({
            'content': [{
                'text': json.dumps({
                    'name': 'jason',
                    'age': 12
                })
            }]
        }).encode()
    }
    
    # Second response with correct uppercase name
    mock_success_response = {
        'body': json.dumps({
            'content': [{
                'text': json.dumps({
                    'name': 'JASON',
                    'age': 12
                })
            }]
        }).encode()
    }

    def side_effect(*args, **kwargs):
        if not hasattr(side_effect, 'called'):
            side_effect.called = True
            return mock_error_response
        return mock_success_response

    mock_bedrock_client.invoke_model.side_effect = side_effect

    response = client.chat.completions.create(
        model=model,
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=3,
    )
    assert response.name == "JASON"


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_upper_case_tenacity(model, mode, mock_bedrock_client):
    """Test retry functionality using tenacity configuration."""
    client = instructor.from_boto3(mock_bedrock_client, mode=mode)
    from tenacity import Retrying, stop_after_attempt, wait_fixed

    # Configure tenacity retry behavior
    retries = Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    )

    # Setup mock responses
    mock_error_response = {
        'body': json.dumps({
            'content': [{
                'text': json.dumps({
                    'name': 'jason',
                    'age': 12
                })
            }]
        }).encode()
    }
    
    mock_success_response = {
        'body': json.dumps({
            'content': [{
                'text': json.dumps({
                    'name': 'JASON',
                    'age': 12
                })
            }]
        }).encode()
    }

    def side_effect(*args, **kwargs):
        if not hasattr(side_effect, 'called'):
            side_effect.called = True
            return mock_error_response
        return mock_success_response

    mock_bedrock_client.invoke_model.side_effect = side_effect

    response = client.chat.completions.create(
        model=model,
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=retries,
    )
    assert response.name == "JASON"

# TODO(#issue): Async retry tests are currently broken due to issues with AsyncMock configuration
# Need to investigate proper way to mock Bedrock async responses
# @pytest.mark.parametrize("model, mode", product(models, modes))
# @pytest.mark.asyncio
# async def test_upper_case_async(model, mode, mock_bedrock_client):
#     """Test basic async retry functionality with uppercase validation."""
#     # Create a simple mock that returns the expected responses d
#     mock = AsyncMock()
#     mock.side_effect = [
#         {
#             'body': json.dumps({
#                 'content': [{
#                     'text': json.dumps({
#                         'name': 'jason',
#                         'age': 12
#                     })
#                 }]
#             }).encode()
#         },
#         {
#             'body': json.dumps({
#                 'content': [{
#                     'text': json.dumps({
#                         'name': 'JASON',
#                         'age': 12
#                     })
#                 }]
#             }).encode()
#         }
#     ]
#     
#     # Attach the mock to the client
#     mock_bedrock_client.ainvoke_model = mock
#     
#     # Create the instructor client
#     client = instructor.from_boto3(mock_bedrock_client, mode=mode)

#     response = await client.chat.completions.create(
#         model=model,
#         response_model=UserDetail,
#         messages=[
#             {"role": "user", "content": "Extract `jason is 12`"},
#         ],
#         max_retries=3,
#     )
#     assert response.name == "JASON"

# @pytest.mark.parametrize("model, mode", product(models, modes))
# @pytest.mark.asyncio
# async def test_upper_case_tenacity_async(model, mode, mock_bedrock_client):
#     """Test async retry functionality using tenacity configuration."""
#     mock = AsyncMock()
#     mock_bedrock_client.ainvoke_model = mock
#     client = instructor.from_boto3(mock_bedrock_client, mode=mode)

#     retries = Retrying(
#         stop=stop_after_attempt(2),
#         wait=wait_fixed(1),
#         retry=retry_if_exception_type(ValidationError)
#     )

#     # Configure mock to return the actual response values
#     mock.side_effect = [mock_error_response, mock_success_response]

#     response = await client.chat.completions.create(
#         model=model,
#         response_model=UserDetail,
#         messages=[
#             {"role": "user", "content": "Extract `jason is 12`"},
#         ],
#         max_retries=retries,
#     )
#     assert response.name == "JASON"