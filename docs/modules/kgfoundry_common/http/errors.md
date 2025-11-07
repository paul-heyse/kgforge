# kgfoundry_common.http.errors

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/errors.py)

## Hierarchy

- **Parent:** [kgfoundry_common.http](../http.md)

## Sections

- **Public API**

## Contents

### kgfoundry_common.http.errors.HttpConnectionError

::: kgfoundry_common.http.errors.HttpConnectionError

*Bases:* HttpError

### kgfoundry_common.http.errors.HttpError

::: kgfoundry_common.http.errors.HttpError

*Bases:* Exception

### kgfoundry_common.http.errors.HttpRateLimited

::: kgfoundry_common.http.errors.HttpRateLimited

*Bases:* HttpStatusError

### kgfoundry_common.http.errors.HttpRequestError

::: kgfoundry_common.http.errors.HttpRequestError

*Bases:* HttpError

### kgfoundry_common.http.errors.HttpStatusError

::: kgfoundry_common.http.errors.HttpStatusError

*Bases:* HttpError

### kgfoundry_common.http.errors.HttpTimeout

::: kgfoundry_common.http.errors.HttpTimeout

*Bases:* HttpError

### kgfoundry_common.http.errors.HttpTlsError

::: kgfoundry_common.http.errors.HttpTlsError

*Bases:* HttpError

### kgfoundry_common.http.errors.HttpTooManyRedirects

::: kgfoundry_common.http.errors.HttpTooManyRedirects

*Bases:* HttpError

## Autorefs Examples

- [kgfoundry_common.http.errors.HttpConnectionError][]
- [kgfoundry_common.http.errors.HttpError][]
- [kgfoundry_common.http.errors.HttpRateLimited][]

## Inheritance

```mermaid
classDiagram
    class HttpConnectionError
    class HttpError
    HttpError <|-- HttpConnectionError
    class HttpError_1
    class Exception
    Exception <|-- HttpError_1
    class HttpRateLimited
    class HttpStatusError
    HttpStatusError <|-- HttpRateLimited
    class HttpRequestError
    HttpError <|-- HttpRequestError
    class HttpStatusError_1
    HttpError <|-- HttpStatusError_1
    class HttpTimeout
    HttpError <|-- HttpTimeout
    class HttpTlsError
    HttpError <|-- HttpTlsError
    class HttpTooManyRedirects
    HttpError <|-- HttpTooManyRedirects
```

## Neighborhood

```d2
direction: right
"kgfoundry_common.http.errors": "kgfoundry_common.http.errors" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/errors.py" }
"kgfoundry_common.http": "kgfoundry_common.http" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/__init__.py" }
"kgfoundry_common.http" -> "kgfoundry_common.http.errors" { style: dashed }
```

