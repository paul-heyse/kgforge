# src/kgfoundry_common/http/errors.py
class HttpError(Exception): ...


class HttpStatusError(HttpError):
    def __init__(
        self, status: int, body_excerpt: str | None = None, headers: dict[str, str] | None = None
    ):
        super().__init__(f"HTTP {status}: {body_excerpt or ''}")
        self.status = status
        self.headers = headers or {}


class HttpRateLimited(HttpStatusError): ...


class HttpTimeout(HttpError): ...


class HttpConnectionError(HttpError): ...


class HttpTlsError(HttpError): ...


class HttpTooManyRedirects(HttpError): ...


class HttpRequestError(HttpError): ...
