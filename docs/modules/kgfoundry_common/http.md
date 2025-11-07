# kgfoundry_common.http

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/__init__.py)

## Hierarchy

- **Parent:** [kgfoundry_common](../kgfoundry_common.md)
- **Children:** [kgfoundry_common.http.errors](http/errors.md), [kgfoundry_common.http.policy](http/policy.md), [kgfoundry_common.http.tenacity_retry](http/tenacity_retry.md), [kgfoundry_common.http.try](http/try.md)

## Sections

- **Public API**

## Contents

### kgfoundry_common.http.make_client_with_policy

::: kgfoundry_common.http.make_client_with_policy

## Relationships

**Imports:** `kgfoundry_common.http.client.HttpClient`, `kgfoundry_common.http.client.HttpSettings`, `kgfoundry_common.http.policy.PolicyRegistry`, `kgfoundry_common.http.tenacity_retry.TenacityRetryStrategy`, `pathlib.Path`

## Autorefs Examples

- [kgfoundry_common.http.make_client_with_policy][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.http": "kgfoundry_common.http" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/__init__.py" }
"kgfoundry_common.http.client.HttpClient": "kgfoundry_common.http.client.HttpClient"
"kgfoundry_common.http" -> "kgfoundry_common.http.client.HttpClient"
"kgfoundry_common.http.client.HttpSettings": "kgfoundry_common.http.client.HttpSettings"
"kgfoundry_common.http" -> "kgfoundry_common.http.client.HttpSettings"
"kgfoundry_common.http.policy.PolicyRegistry": "kgfoundry_common.http.policy.PolicyRegistry"
"kgfoundry_common.http" -> "kgfoundry_common.http.policy.PolicyRegistry"
"kgfoundry_common.http.tenacity_retry.TenacityRetryStrategy": "kgfoundry_common.http.tenacity_retry.TenacityRetryStrategy"
"kgfoundry_common.http" -> "kgfoundry_common.http.tenacity_retry.TenacityRetryStrategy"
"pathlib.Path": "pathlib.Path"
"kgfoundry_common.http" -> "pathlib.Path"
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.http" { style: dashed }
"kgfoundry_common.http.errors": "kgfoundry_common.http.errors" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/errors.py" }
"kgfoundry_common.http" -> "kgfoundry_common.http.errors" { style: dashed }
"kgfoundry_common.http.policy": "kgfoundry_common.http.policy" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/policy.py" }
"kgfoundry_common.http" -> "kgfoundry_common.http.policy" { style: dashed }
"kgfoundry_common.http.tenacity_retry": "kgfoundry_common.http.tenacity_retry" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/tenacity_retry.py" }
"kgfoundry_common.http" -> "kgfoundry_common.http.tenacity_retry" { style: dashed }
"kgfoundry_common.http.try": "kgfoundry_common.http.try" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/try.py" }
"kgfoundry_common.http" -> "kgfoundry_common.http.try" { style: dashed }
```

