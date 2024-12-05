from __future__ import annotations

from typing import Any, overload

import boto3

import instructor
from instructor.utils import Provider


@overload
def from_boto3(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BOTO3_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    ...


def from_boto3(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BOTO3_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    """Create an Instructor instance from a Boto3 client.

    Args:
        client: An instance of Boto3 client
        mode: The mode to use for the client (BOTO3_JSON or BOTO3_TOOLS)
        use_async: Whether to use the async version (not applicable for boto3)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance

    Raises:
        TypeError: If client is not a Boto3 client instance
        AssertionError: If mode is not BOTO3_JSON or BOTO3_TOOLS
    """
    assert mode in {
        instructor.Mode.BOTO3_JSON,
        instructor.Mode.BOTO3_TOOLS,
    }, "Mode must be one of {instructor.Mode.BOTO3_JSON, instructor.Mode.BOTO3_TOOLS}"

    if not hasattr(client, 'invoke_model'):
        raise TypeError("Client must be a Boto3 client instance")

    # Wrap the client's invoke_model method with the appropriate mode
    create = instructor.patch(create=client.invoke_model, mode=mode)

    return instructor.Instructor(
        client=client,
        create=create,
        provider=Provider.BOTO3,
        mode=mode,
        **kwargs,
    )

def create_tool_call(self, name: str, arguments: str) -> dict:
    """Create a tool call object in Bedrock format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments
        }
    }

def format_tool_response(self, content: str) -> dict:
    """Format tool response in Bedrock format."""
    return {
        "role": "function",
        "content": content
    }