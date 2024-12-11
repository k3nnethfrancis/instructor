from __future__ import annotations

from typing import Any, Optional, Type, TypeVar, Callable, Tuple, Iterator
from pydantic import BaseModel
import json
import base64
from functools import partial
import instructor
from dataclasses import dataclass
from collections.abc import Iterable
from instructor.exceptions import InstructorError, InstructorRetryException, InstructorBedrock
from tenacity import Retrying, RetryCallState, retry_if_exception_type
from pydantic import ValidationError
from pydantic import create_model
from instructor.process_response import handle_boto3_tools
from botocore.exceptions import ClientError

T = TypeVar("T", bound=BaseModel)

def handle_validation_error(retry_state: RetryCallState) -> None:
    """Handle validation errors during retries."""
    if retry_state.outcome is not None and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        if isinstance(exc, ValidationError):
            raise InstructorRetryException(
                str(exc),
                last_completion=retry_state.outcome.exception(),
                n_attempts=retry_state.attempt_number,
                total_usage=0,  # Bedrock doesn't provide usage info
            )
        raise exc

def handle_stream_response(response: dict, response_model: Type[T]) -> Iterator[T]:
    """Handle streaming response from Bedrock."""
    # Get the actual model type if it's an Iterable
    if hasattr(response_model, "__origin__") and response_model.__origin__ is Iterable:
        actual_model = response_model.__args__[0]
    else:
        actual_model = response_model

    # Check if this is a partial model and get the base model
    is_partial = hasattr(actual_model, '__instructor_is_partial__')
    base_model = actual_model.__pydantic_generic_metadata__['args'][0] if is_partial else actual_model

    for chunk in response['body']:
        # Decode base64 chunk
        chunk_data = json.loads(base64.b64decode(chunk['chunk']['bytes']))
        if chunk_data['type'] == 'content_block_delta':
            try:
                content = json.loads(chunk_data['delta']['text'])
                # Create a model instance with partial data
                model_instance = base_model.model_construct(**content)
                model_instance._raw_response = chunk_data
                yield model_instance
            except ValidationError as e:
                if not is_partial:
                    raise InstructorError(
                        original_exception=e,
                        response=chunk_data,
                        model=actual_model,
                    )

def _create(
    client: Any,
    response_model: Type[T],
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> T:
    """Create a completion using the Bedrock client."""
    # Get the processed model and kwargs from the handler
    processed_model, processed_kwargs = handle_boto3_tools(response_model, {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    })

    # Define models that support tool use
    TOOL_SUPPORTED_MODELS = {
        "cohere.command-r-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        # Add other models as they become available
    }

    try:
        # Check if model supports tools before using converse
        if "toolConfig" in processed_kwargs:
            if model not in TOOL_SUPPORTED_MODELS:
                # Fall back to regular invoke_model if tools not supported
                del processed_kwargs["toolConfig"]
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    **processed_kwargs
                }
                response = client.invoke_model(
                    modelId=model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
            else:
                response = client.converse(
                    modelId=model,
                    messages=processed_kwargs["messages"],
                    toolConfig=processed_kwargs["toolConfig"],
                )
        else:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                **processed_kwargs
            }
            response = client.invoke_model(
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

        return _process_response(response, response_model)
        
    except ClientError as e:
        if "ValidationException" in str(e):
            raise InstructorBedrock(e)
        raise InstructorRetryException(
            str(e),
            last_completion=e,
            n_attempts=1,
            total_usage=0,
        )
    except Exception as e:
        raise InstructorRetryException(
            str(e),
            last_completion=e,
            n_attempts=1,
            total_usage=0,
        )

def create(
    client: Any,
    response_model: Type[T],
    messages: list[dict[str, str]],
    model: str = "cohere.command-r-v1:0",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs: Any,
) -> T | Iterator[T]:
    """Create a completion using the Bedrock client."""
    max_retries = kwargs.pop('max_retries', None)

    # Handle retries
    if max_retries is not None:
        if isinstance(max_retries, int):
            retrying = Retrying(
                stop=lambda retry_state: retry_state.attempt_number > max_retries,
                retry=retry_if_exception_type(InstructorRetryException),  # Only retry RetryExceptions
                reraise=True,
            )
        else:
            retrying = max_retries
        return retrying(lambda: _create(
            client=client,
            response_model=response_model,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ))

    return _create(
        client=client,
        response_model=response_model,
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

def create_with_completion(
    client: Any,
    response_model: Type[T],
    messages: list[dict[str, str]],
    # model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    model: str = "anthropic.claude-v2",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[T, dict]:
    """Create a completion and return both the model and raw response."""
    model_instance = create(
        client=client,
        response_model=response_model,
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    return model_instance, model_instance._raw_response

def _process_response(response: dict, response_model: Type[T]) -> T:
    """Process the response from Bedrock."""
    try:
        # Extract content from response
        if 'output' in response:
            # Handle converse endpoint response
            message = response['output']['message']
            if message['role'] == 'assistant':
                content = message['content']
                # Look for toolUse in the content
                for item in content:
                    if 'toolUse' in item:
                        tool_use = item['toolUse']
                        if 'input' in tool_use:
                            data = tool_use['input']
                            # Parse any string values that should be dictionaries
                            for key, value in data.items():
                                if isinstance(value, str) and value.startswith('{'):
                                    try:
                                        data[key] = json.loads(value)
                                    except json.JSONDecodeError:
                                        pass  # Keep original value if not valid JSON
                            
                            # Create model instance
                            model_instance = response_model.model_validate(data)
                            model_instance._raw_response = response
                            return model_instance
                
                # If no toolUse found, try the text content
                text_content = content[0]['text']
                try:
                    data = json.loads(text_content)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', text_content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        raise ValueError(f"Could not extract JSON from response: {text_content}")

        else:
            # Handle invoke_model endpoint response
            body = response.get('body')
            if hasattr(body, 'read'):
                content = body.read().decode('utf-8')
                data = json.loads(content)
            else:
                raise ValueError(f"Unexpected response format: {response}")

        # Create model instance
        model_instance = response_model.model_validate(data)
        model_instance._raw_response = response
        return model_instance

    except Exception as e:
        raise InstructorError(
            f"Error processing response: {str(e)}",
            original_exception=e,
            response=response,
            model=response_model
        )

@dataclass
class Completions:
    """Completions interface matching OpenAI's structure."""
    create: Callable
    create_with_completion: Callable

@dataclass
class Chat:
    """Chat interface matching OpenAI's structure."""
    completions: Completions

@dataclass
class Client:
    """Client interface matching OpenAI's structure."""
    chat: Chat

def from_boto3(client: Any, mode: instructor.Mode = instructor.Mode.BOTO3_TOOLS, **kwargs: Any) -> Client:
    """Create a patched client for Bedrock."""
    create_fn = partial(create, client)
    create_with_completion_fn = partial(create_with_completion, client)
    
    completions = Completions(
        create=create_fn,
        create_with_completion=create_with_completion_fn
    )
    chat = Chat(completions=completions)
    return Client(chat=chat)