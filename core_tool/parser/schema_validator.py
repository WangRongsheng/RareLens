"""Validate parsed results against a schema (Pydantic)."""

from __future__ import annotations

import logging
from typing import Any, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def validate(data: dict[str, Any], schema_cls: Type[BaseModel]) -> BaseModel:
    try:
        return schema_cls.model_validate(data)
    except ValidationError as e:
        logger.warning("Schema validation failed for %s: %s", schema_cls.__name__, e)
        raise
