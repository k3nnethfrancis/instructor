from itertools import product
from collections.abc import Iterable
from typing import Iterator
from unittest.mock import Mock
from pydantic import BaseModel
import pytest
import instructor
from instructor import Mode
import logging
import base64
import json

logger = logging.getLogger(__name__)

class UserExtract(BaseModel):
    name: str
    age: int

def test_stream_response(mock_bedrock_client):
    """Test streaming response handling for Bedrock."""
    logging.basicConfig(level=logging.DEBUG)
    
    def streaming_body() -> Iterator[dict]:
        chunks = [
            {
                'chunk': {
                    'bytes': base64.b64encode(
                        json.dumps({
                            'type': 'content_block_delta',
                            'delta': {
                                'type': 'text_delta',
                                'text': '{"name": "Alice", "age": 25}'
                            },
                            'index': 0
                        }).encode('utf-8')
                    ).decode('utf-8')
                }
            },
            {
                'chunk': {
                    'bytes': base64.b64encode(
                        json.dumps({
                            'type': 'content_block_delta',
                            'delta': {
                                'type': 'text_delta',
                                'text': '{"name": "Bob", "age": 30}'
                            },
                            'index': 1
                        }).encode('utf-8')
                    ).decode('utf-8')
                }
            }
        ]
        for chunk in chunks:
            yield chunk

    mock_response = {'body': streaming_body()}
    mock_bedrock_client.invoke_model_with_response_stream = Mock(return_value=mock_response)
    
    client = instructor.from_boto3(mock_bedrock_client, mode=Mode.BOTO3_TOOLS)
    
    response = client.chat.completions.create(
        model="anthropic.claude-v2",
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )

    for m in response:
        assert isinstance(m, UserExtract)