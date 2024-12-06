from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel
import pytest
import instructor
from instructor.dsl.partial import Partial
import json
import base64
from instructor import Mode

# Test configuration
models = ["anthropic.claude-v2"]
modes = [Mode.BOTO3_JSON, Mode.BOTO3_TOOLS]


class UserExtract(BaseModel):
    name: str
    age: int


def create_stream_chunk(content: dict) -> dict:
    """Helper to create properly formatted stream chunks."""
    return {
        'chunk': {
            'bytes': base64.b64encode(
                json.dumps({
                    'type': 'content_block_delta',
                    'delta': {
                        'type': 'text_delta',
                        'text': json.dumps(content)
                    }
                }).encode('utf-8')
            ).decode('utf-8')
        }
    }


@pytest.mark.parametrize("model, mode, stream", product(models, modes, [True, False]))
def test_iterable_model(model, mode, stream, mock_bedrock_client):
    client = instructor.from_boto3(mock_bedrock_client, mode=mode)
    
    # Setup mock response based on stream parameter
    if stream:
        chunks = [
            create_stream_chunk({"name": "Alice", "age": 25}),
            create_stream_chunk({"name": "Bob", "age": 30})
        ]
        mock_bedrock_client.invoke_model_with_response_stream.return_value = {
            'body': iter(chunks)
        }
    else:
        # The non-streaming response should be a single JSON object
        mock_bedrock_client.invoke_model.return_value = {
            'body': json.dumps({
                'content': [
                    {
                        'text': json.dumps({
                            "name": "Alice",
                            "age": 25
                        })
                    },
                    {
                        'text': json.dumps({
                            "name": "Bob",
                            "age": 30
                        })
                    }
                ]
            }).encode()
        }

    response = client.chat.completions.create(
        model=model,
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=stream,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    
    for m in response:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_partial_model(model, mode, mock_bedrock_client):
    client = instructor.from_boto3(mock_bedrock_client, mode=mode)
    
    chunks = [
        create_stream_chunk({"name": "Kenneth Francis", "age": 99}),
        create_stream_chunk({"name": "Kenneth Francis"})
    ]
    mock_bedrock_client.invoke_model_with_response_stream.return_value = {
        'body': iter(chunks)
    }

    response = client.chat.completions.create(
        model=model,
        response_model=Partial[UserExtract],
        max_retries=2,
        max_tokens=1024,
        stream=True,
        messages=[
            {"role": "user", "content": "Kenneth Francis is 99 years old"},
        ],
    )
    
    for m in response:
        assert isinstance(m, UserExtract)


@pytest.mark.parametrize("model,mode", product(models, modes))
def test_model(model, mode, mock_bedrock_client):
    client = instructor.from_boto3(mock_bedrock_client, mode=mode)
    
    mock_bedrock_client.invoke_model.return_value = {
        'body': json.dumps({
            'content': [{
                'text': json.dumps({
                    "name": "Kenneth",
                    "age": 99
                })
            }]
        }).encode()
    }

    response = client.chat.completions.create(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Kenneth is 99 years old"}],
    )

    assert response.name == "Kenneth"
    assert response.age == 99
