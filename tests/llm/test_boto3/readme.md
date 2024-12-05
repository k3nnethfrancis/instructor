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

## Review Notes
This implementation complements the existing Bedrock support by adding direct boto3 integration, giving users more flexibility in how they interact with AWS services.

### Key Files Modified
```
instructor/
├── __init__.py          # Added from_boto3
├── mode.py             # Added BOTO3 modes
├── client_boto3.py     # New file
└── function_calls.py   # Updated for boto3 support
```

### Testing Locally
1. Configure AWS credentials
2. Install test dependencies: `pip install -r requirements-test.txt`
3. Run tests: `pytest tests/llm/test_boto3/`