# Boto3 Direct Integration for AWS Bedrock

## Context
While Bedrock support exists through the Anthropic client ([Issue #540](https://github.com/instructor-ai/instructor/issues/540)), this implementation adds direct `boto3` support via `instructor.from_boto3()`. This allows users to work directly with boto3 clients while maintaining Instructor's structured output capabilities.

## Implementation Details

### New Features
- Direct boto3 client support via `instructor.from_boto3()`
- Two dedicated modes:
  ```python
  instructor.Mode.BOTO3_TOOLS  # For function calling
  instructor.Mode.BOTO3_JSON   # For direct JSON responses
  ```
- Streaming support with proper chunk handling
- Retry logic matching AWS best practices

### Response Format Handling
Implemented specific handling for Bedrock's response format:
```python
{
    'chunk': {
        'bytes': base64.b64encode(
            json.dumps({
                'type': 'content_block_delta',
                'delta': {
                    'type': 'text_delta',
                    'text': 'actual_content'
                }
            }).encode('utf-8')
        ).decode('utf-8')
    }
}
```

## Test Coverage

### Core Tests
- `test_simple.py`: Basic response handling and mode switching
- `test_stream.py`: Streaming response parsing
- `test_retries.py`: Error handling and retry logic

### Verified Functionality
- ✅ Basic function calling
- ✅ Streaming responses
- ✅ Error handling and retries
- ✅ Base64 decoding
- ✅ JSON parsing

## Example Usage
```python
import boto3
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Initialize and patch the client
bedrock = boto3.client('bedrock-runtime')
client = instructor.from_boto3(bedrock)

# Get structured response
response = client.invoke_model(
    modelId="anthropic.claude-v2",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "prompt": "\n\nHuman: Extract: Alice is 30 years old\n\nAssistant:",
        "max_tokens_to_sample": 1000,
        "temperature": 0.7,
        "anthropic_version": "bedrock-2023-05-31"
    })
)
```

## Review Notes
- Implementation follows existing client patterns
- Maintains compatibility with boto3's native interface
- Adds structured output capabilities to Bedrock
- Includes comprehensive test coverage
- Provides working examples in `examples/boto3_bedrock/`

## Testing Locally
1. Configure AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_DEFAULT_REGION="us-west-2"
   ```
2. Install test dependencies: `pip install -r requirements-test.txt`
3. Run tests: `pytest tests/llm/test_boto3/`
4. Try example: `python -m examples.boto3_bedrock.run`