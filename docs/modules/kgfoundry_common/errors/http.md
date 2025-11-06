# kgfoundry_common.errors.http

HTTP adapters for Problem Details exception handling.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/kgfoundry_common/errors/http.py)

## Sections

- **Public API**

## Contents

### kgfoundry_common.errors.http.problem_details_response

::: kgfoundry_common.errors.http.problem_details_response

### kgfoundry_common.errors.http.register_problem_details_handler

::: kgfoundry_common.errors.http.register_problem_details_handler

## Relationships

**Imports:** `__future__.annotations`, `asyncio`, `fastapi.FastAPI`, `fastapi.Request`, `fastapi.responses.JSONResponse`, `kgfoundry_common.errors.exceptions.KgFoundryError`, `kgfoundry_common.fastapi_helpers.typed_exception_handler`, `kgfoundry_common.logging.get_logger`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.problem_details.ProblemDetails`, `typing.TYPE_CHECKING`

## Autorefs Examples

- [kgfoundry_common.errors.http.problem_details_response][]
- [kgfoundry_common.errors.http.register_problem_details_handler][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.errors.http": "kgfoundry_common.errors.http" { link: "./kgfoundry_common/errors/http.md" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.errors.http" -> "__future__.annotations"
"asyncio": "asyncio"
"kgfoundry_common.errors.http" -> "asyncio"
"fastapi.FastAPI": "fastapi.FastAPI"
"kgfoundry_common.errors.http" -> "fastapi.FastAPI"
"fastapi.Request": "fastapi.Request"
"kgfoundry_common.errors.http" -> "fastapi.Request"
"fastapi.responses.JSONResponse": "fastapi.responses.JSONResponse"
"kgfoundry_common.errors.http" -> "fastapi.responses.JSONResponse"
"kgfoundry_common.errors.exceptions.KgFoundryError": "kgfoundry_common.errors.exceptions.KgFoundryError"
"kgfoundry_common.errors.http" -> "kgfoundry_common.errors.exceptions.KgFoundryError"
"kgfoundry_common.fastapi_helpers.typed_exception_handler": "kgfoundry_common.fastapi_helpers.typed_exception_handler"
"kgfoundry_common.errors.http" -> "kgfoundry_common.fastapi_helpers.typed_exception_handler"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"kgfoundry_common.errors.http" -> "kgfoundry_common.logging.get_logger"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.errors.http" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.problem_details.ProblemDetails": "kgfoundry_common.problem_details.ProblemDetails"
"kgfoundry_common.errors.http" -> "kgfoundry_common.problem_details.ProblemDetails"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"kgfoundry_common.errors.http" -> "typing.TYPE_CHECKING"
"kgfoundry_common.errors.http_code": "kgfoundry_common.errors.http code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/kgfoundry_common/errors/http.py" }
"kgfoundry_common.errors.http" -> "kgfoundry_common.errors.http_code" { style: dashed }
```

