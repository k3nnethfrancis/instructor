import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_bedrock_client():
    """Create a mocked Bedrock client."""
    mock_client = MagicMock()
    # Ensure the client has the 'invoke_model' method
    mock_client.invoke_model = MagicMock()

    return mock_client