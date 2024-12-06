from __future__ import annotations

from typing import Any, Optional, Type
from pydantic import BaseModel
from botocore.exceptions import ClientError


class IncompleteOutputException(Exception):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)


class InstructorRetryException(Exception):
    def __init__(
        self,
        *args: list[Any],
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: int,
        create_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        self.messages = messages
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        self.create_kwargs = create_kwargs
        super().__init__(*args, **kwargs)


class InstructorError(Exception):
    """Base class for all instructor errors."""

    def __init__(
        self,
        message: str = "",
        original_exception: Exception | None = None,
        response: Any = None,
        model: Any = None,
    ) -> None:
        self.original_exception = original_exception
        self.response = response
        self.model = model
        super().__init__(message or str(original_exception))


class InstructorBedrock(InstructorError):
    """Bedrock-specific error handling."""
    def __init__(
        self,
        client_error: ClientError,
        response: Any = None,
        model: Any = None,
    ) -> None:
        message = client_error.response['Error']['Message']
        super().__init__(
            message=message,
            original_exception=client_error,
            response=response,
            model=model,
        )


class InstructorRetryException(InstructorError):
    """Exception raised when a retry is needed."""

    def __init__(
        self,
        message: str,
        *,
        last_completion: Any = None,
        n_attempts: int = 0,
        total_usage: int = 0,
    ) -> None:
        super().__init__(message)
        self.last_completion = last_completion
        self.n_attempts = n_attempts
        self.total_usage = total_usage
