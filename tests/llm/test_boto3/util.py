"""Utility functions for Boto3 tests."""

def create_mock_response(content: dict) -> dict:
    """Create a mock Bedrock response with proper structure."""
    return {
        'contentType': 'application/json',
        'body': content
    }