"""Check SubQ model availability without executing a billable model request."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODELS_URL = "https://api.subq.ai/v1/models"
TARGET_MODEL = "subq-savant-1.4"
TIMEOUT_SECONDS = 10


class AccessStatus(str, Enum):
    AVAILABLE = "available"
    NOT_LISTED = "not_listed"
    MISSING_KEY = "missing_key"
    AUTH_FAILED = "auth_failed"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class AccessResult:
    status: AccessStatus
    model: str = TARGET_MODEL
    models_seen: int = 0
    http_status: int | None = None


def check_model_access(
    api_key: str | None,
    *,
    open_url: Callable[..., object] = urlopen,
) -> AccessResult:
    """Use only GET /v1/models; no model inference or paid requests.

    The injected opener supports tests without touching the network.
    """
    if not api_key or not api_key.strip():
        return AccessResult(AccessStatus.MISSING_KEY)

    request = Request(
        MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with open_url(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as error:
        if error.code in (401, 403):
            return AccessResult(AccessStatus.AUTH_FAILED, http_status=error.code)
        return AccessResult(AccessStatus.HTTP_ERROR, http_status=error.code)
    except (URLError, TimeoutError, OSError):
        return AccessResult(AccessStatus.NETWORK_ERROR)
    except (json.JSONDecodeError, UnicodeError, ValueError):
        return AccessResult(AccessStatus.INVALID_RESPONSE)

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return AccessResult(AccessStatus.INVALID_RESPONSE)
    rows = payload["data"]
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
        return AccessResult(AccessStatus.INVALID_RESPONSE)

    available = TARGET_MODEL in {row["id"] for row in rows}
    return AccessResult(
        AccessStatus.AVAILABLE if available else AccessStatus.NOT_LISTED,
        models_seen=len(rows),
    )


def main() -> int:
    result = check_model_access(os.getenv("SUBQ_API_KEY"))
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.status == AccessStatus.AVAILABLE else 2


if __name__ == "__main__":
    raise SystemExit(main())
