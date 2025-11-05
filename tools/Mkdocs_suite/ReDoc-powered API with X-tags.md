Awesome—let’s make your **ReDoc-powered API area** feel first-class and tightly woven into the rest of your docs, diagrams, and code pages (not a bolt-on “API island”). Below is a complete, junior-friendly plan with guidelines, examples, and build scripts.

---

# What “best-in-class ReDoc” looks like

1. **A tuned ReDoc page** embedded in MkDocs, with:

   * clean tag taxonomy (+ **x-tagGroups** for 2-level nav)
   * stable, URL-safe **operationId**s (for deep links from everywhere)
   * rich samples (**x-codeSamples**), meaningful **examples**, and optional **x-badges**
   * tasteful config (expand levels, sorting, path in middle panel, etc.). ([Redocly][1])

2. **API “tags as a first-class navigation concept”** across the whole site:

   * per-tag pages in MkDocs (not just in ReDoc)
   * **clickable tag chips** on module pages (Material’s tags plugin) that jump into the ReDoc operation section
   * **diagrams** that show API tags and operations; nodes click straight into ReDoc. ([GitHub Pages][2])

3. **Source↔API cross-links**:

   * from code/module pages → ReDoc **#operation/{operationId}** anchors
   * from ReDoc (via **externalDocs** and/or tag descriptions) → relevant module or architecture pages
   * optional private vendor extension (e.g., **x-handler**) that records the code path that implements an operation (so generators can wire links automatically). ([Redocly][3])

4. **API quality gate in CI**:

   * enforce tag hygiene, operationId uniqueness + URL-safety, and descriptions with **Redocly CLI** rules. ([Redocly][4])

---

# A. Embed ReDoc in MkDocs (two solid choices)

### Option 1 — Use `mkdocs-redoc-tag` (simplest)

In any Markdown page (e.g., `docs/api/index.md`):

```md
---
hide:
  - navigation
  - toc
---

# HTTP API

<redoc src="../../openapi/openapi.yaml"/>
```

And in `mkdocs.yml`:

```yaml
plugins:
  - redoc-tag
```

This plugin ships the ReDoc assets with your site (no CDN), syncs dark mode with Material, and works offline—nice for intranet docs. (It uses an iframe; you don’t pass many runtime options.) ([GitHub][5])

### Option 2 — Prebuild a ReDoc HTML with options (more control)

If you need specific ReDoc **options** (e.g., `pathInMiddlePanel`, `jsonSamplesExpandLevel`, `scrollYOffset`, `hideDownloadButtons`, etc.), prebuild a static page:

```bash
# install once
npm i -g @redocly/cli
# generate a standalone redoc page
redocly build-docs openapi/openapi.yaml --output docs/api/reference.html
```

Then link to `docs/api/reference.html` from your nav or embed it in an `<iframe>`. ReDoc options are extensive—sorting, expansion, samples, and layout—and differ between CE 2.x vs newer docs; see the config reference for the canonical list and kebab-case attribute names. ([Redocly][1])

**Recommended ReDoc options for big specs**

* `pathInMiddlePanel: true` (move method+path into center panel)
* `jsonSamplesExpandLevel: 3..5` (or `"all"` during authoring)
* `schemasExpansionLevel: 1..2`
* `sortTagsAlphabetically: true`, `sortOperationsAlphabetically: true`
* `scrollYOffset: ".md-header"` (so deep links account for sticky header)
* `hideDownloadButtons: true` (if you don’t want spec download)
  All are documented in the ReDoc config reference. ([Redocly][1])

---

# B. Design a tag taxonomy that works across ReDoc *and* MkDocs

## 1) Authoritative tags (OpenAPI)

Define **tags** at the root with clear descriptions and (optional) `externalDocs` pointing back into your MkDocs content:

```yaml
tags:
  - name: orders
    x-displayName: Orders & Checkout
    description: Endpoints to create and manage orders.
    externalDocs:
      description: Architecture notes for the ordering domain
      url: /architecture/orders/           # MkDocs page
  - name: auth
    x-displayName: Authentication
    description: Login, refresh, and revoke tokens.
```

Use **one tag per operation** when tags act as categories; it keeps nav clean. Redocly rules can enforce this. **x-displayName** gives you a pretty label without changing the canonical `name`. ([Redocly][6])

Group your tags with **x-tagGroups** for a two-level left nav in ReDoc:

```yaml
x-tagGroups:
  - name: Core
    tags: [orders, customers, auth]
  - name: Admin
    tags: [catalog, pricing]
```

ReDoc will render groups → tags → operations; any tag *not* in a group won’t show—so list them all. ([Redocly][7])

## 2) Traits (non-nav markers)

Use **x-traitTag: true** for traits like *Pagination* or *Rate Limits*. ReDoc shows them as info (no sub-items) and renders the description, which is great for shared concerns. ([GitHub][8])

## 3) Stable deep links

Every operation needs a **globally unique, URL-safe `operationId`** so you can deep link to it from module pages, diagrams, and blog posts:

```yaml
paths:
  /orders:
    post:
      operationId: orders.create
      tags: [orders]
```

Redocly CLI rules to enforce this:

```yaml
# redocly.yaml
extends:
  - recommended
rules:
  operation-operationId: error
  operation-operationId-unique: error
  operation-operationId-url-safe: error
  operation-tag-defined: error
  tag-description: warn
```

Run `redocly lint openapi/openapi.yaml` in CI to keep it healthy. ([Redocly][9])

---

# C. Add depth to operations (samples, badges, logo)

* **x-codeSamples** — language-specific snippets shown in the right panel:

```yaml
paths:
  /orders:
    post:
      operationId: orders.create
      tags: [orders]
      x-codeSamples:
        - lang: Python
          source: |
            client.create_order({"sku": "ABC", "qty": 2})
        - lang: curl
          source: |
            curl -X POST /orders -d '{"sku":"ABC","qty":2}'
```

* **examples / x-examples** — structured request/response examples to show realistic payloads.
* **x-badges** — add small capsules like “Beta” or “Deprecated” to the operation.
* **x-logo** (under `info`) — tasteful branding at the top of ReDoc. ([Redocly][10])

---

# D. Make APIs visible *everywhere* (generators you can paste)

## 1) Generate **per-tag pages** in MkDocs (clickable into ReDoc)

Add `docs/_scripts/gen_api_tag_pages.py` and wire it in `mkdocs.yml` under `gen-files`.

```python
# docs/_scripts/gen_api_tag_pages.py
from __future__ import annotations
import yaml, mkdocs_gen_files
from pathlib import Path

SPEC = Path("openapi/openapi.yaml")

with SPEC.open() as f:
    oas = yaml.safe_load(f)

# Build tag lookup
tag_meta = {t["name"]: t for t in oas.get("tags", [])}

# Iterate operations
tag_to_ops = {}
for path, item in (oas.get("paths") or {}).items():
    for method, op in (item or {}).items():
        if method.lower() not in {"get","put","post","delete","patch","options","head","trace"}:
            continue
        op_id = op.get("operationId")
        for tag in op.get("tags") or []:
            tag_to_ops.setdefault(tag, []).append({
                "id": op_id, "method": method.upper(), "path": path, "summary": op.get("summary","")
            })

# Write a landing page
with mkdocs_gen_files.open("api/tags/index.md", "w") as f:
    f.write("# API tags\n\n")
    for tag in sorted(tag_to_ops):
        f.write(f"- [{tag}](./{tag}.md)\n")

# Write one page per tag
for tag, ops in sorted(tag_to_ops.items()):
    meta = tag_meta.get(tag, {})
    with mkdocs_gen_files.open(f"api/tags/{tag}.md", "w") as f:
        display = meta.get("x-displayName", tag)
        f.write(f"# {display}\n\n")
        if "description" in meta:
            f.write(meta["description"] + "\n\n")
        if (ext := meta.get("externalDocs")):
            f.write(f"> See also: [{ext.get('description','More…')}]({ext['url']})\n\n")
        for o in sorted(ops, key=lambda v: v["path"]):
            # Deep link to ReDoc section
            f.write(f"- **{o['method']}** `{o['path']}` — {o['summary']} "
                    f"[View in ReDoc](../index/#operation/{o['id']})\n")
```

This gives you **MkDocs pages for each tag** (with summaries and one-click jumps into ReDoc), so tags are a first-class concept outside ReDoc too. ([Redocly][3])

> Put a “Tags” entry in your MkDocs nav under “API”. The `externalDocs` links you define on tags can point to your **Architecture** pages (conceptual guides), closing the loop. ([Swagger][11])

## 2) Put **tag chips** on your module pages (Material tags plugin)

Enable Material’s `tags` plugin and, when you generate your module pages, add front-matter like:

```md
---
tags: [ "api:orders", "layer:domain" ]
---
```

Create a simple mapping in your generator (e.g., `api_usage.json`) that lists the API tags each module touches; write them into the module page’s YAML. Material shows the chips and creates a tags index—clicking the chip lands on `api/tags/orders.md`, which then deep-links into ReDoc. ([GitHub Pages][2])

## 3) Draw **API diagrams** (D2) with two-way navigation

Generate a diagram that clusters operations by tag, with **links to ReDoc**:

```python
# docs/_scripts/gen_api_diagrams.py (append to your D2 generator)
import yaml, mkdocs_gen_files
oas = yaml.safe_load(open("openapi/openapi.yaml"))
ops = []
for p, item in (oas.get("paths") or {}).items():
    for m, op in (item or {}).items():
        if m.upper() in {"GET","PUT","POST","DELETE","PATCH","OPTIONS","HEAD","TRACE"}:
            if op.get("operationId"):
                for tag in op.get("tags") or ["untagged"]:
                    ops.append((tag, m.upper(), p, op["operationId"]))
with mkdocs_gen_files.open("diagrams/api_by_tag.d2", "w") as d:
    d.write('direction: right\nAPIs: "APIs" {\n')
    for tag, *_ in sorted({(t,) for (t,_,_,_) in ops}):
        d.write(f'  "{tag}": "{tag}" {{}}\n')
    for tag, method, path, opid in ops:
        node = f'{method} {path}'
        d.write(f'  "{node}": "{node}" {{ link: "../api/index/#operation/{opid}" }}\n')
        d.write(f'  "{tag}" -> "{node}"\n')
    d.write('}\n')
```

ReDoc accepts deep links of the form `#operation/{operationId}`, so every node in the diagram jumps into the correct operation in ReDoc. ([GitHub][12])

---

# E. Wire **code ↔ API** automatically (for humans *and* agents)

Add a **private vendor extension** to each operation that records the code symbol that implements it:

```yaml
paths:
  /orders:
    post:
      operationId: orders.create
      tags: [orders]
      x-handler: "your_package.orders.service:create_order"
```

Because `x-*` extensions are allowed anywhere in OpenAPI, your generators can read `x-handler` and:

* add “Related API operations” on the module page (with deep links),
* add **tag chips** to the module,
* draw edges from module nodes to API nodes in D2. ([Swagger][13])

> Using FastAPI? You can inject vendor extensions by post-processing the schema or with a custom OpenAPI function. You can also add **x-logo** in the same customizer. ([FastAPI][14])

---

# F. Authoring guidelines (what to teach the team)

**Tags & groups**

* Treat tags as **product categories** (not dumping grounds). Prefer *one tag per operation*; if you need more, use **x-traitTag** (e.g., Pagination). Group tags with **x-tagGroups** (Core, Admin, etc.). Enforce with Redocly rules. ([Redocly][6])

**operationId**

* Required, unique, **URL-safe**, and stable. Use a predictable pattern like `resource.action` (`orders.create`, `orders.list`). Enforce with Redocly rules. Deep links from docs/diagrams/modules rely on this. ([Redocly][15])

**Samples & examples**

* Provide **x-codeSamples** in the languages your users care about (at least cURL + one SDK). Pair with realistic **examples** under request/response content. ([Redocly][10])

**Doc cohesion**

* Use tag-level `externalDocs` to link to your MkDocs **Architecture** pages (domain guides). In those pages, link back into ReDoc via `#operation/{operationId}`. ([Swagger][11])

**Branding & clarity**

* Add **x-logo** under `info` and keep the left nav readable with **x-displayName** on tags. ([Redocly][16])

---

# G. Example: “gold standard” OpenAPI header & one operation

```yaml
openapi: 3.1.0
info:
  title: Example Store API
  version: 1.2.0
  description: >
    REST API for order placement and management.
  x-logo:
    url: https://example.com/logo.svg
    altText: Example
tags:
  - name: orders
    x-displayName: Orders & Checkout
    description: Endpoints to create and manage orders.
    externalDocs:
      description: Domain overview
      url: /architecture/orders/
  - name: auth
    x-displayName: Authentication
    description: Token issuance and revocation.

x-tagGroups:
  - name: Core
    tags: [orders, auth]

paths:
  /orders:
    post:
      operationId: orders.create
      tags: [orders]               # single category tag
      summary: Create an order
      description: Create a new order from items in the cart.
      x-handler: "your_package.orders.service:create_order"
      x-codeSamples:
        - lang: curl
          source: |
            curl -X POST /orders -H 'Content-Type: application/json' \
                 -d '{"items":[{"sku":"ABC","qty":2}]}'
        - lang: Python
          source: |
            client.create_order({"items":[{"sku":"ABC","qty":2}]})
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/NewOrder' }
            examples:
              simple:
                value: { items: [ { sku: "ABC", qty: 2 } ] }
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema: { $ref: '#/components/schemas/Order' }
              examples:
                ok:
                  value: { id: "ord_123", status: "pending" }
components:
  schemas:
    NewOrder: { type: object, properties: { items: { type: array, items: { $ref: '#/components/schemas/Item' } } }, required: [items] }
    Item:     { type: object, properties: { sku: { type: string }, qty: { type: integer, minimum: 1 } }, required: [sku, qty] }
    Order:    { type: object, properties: { id: { type: string }, status: { type: string } }, required: [id, status] }
```

* Notes: `operationId` enables ReDoc deep links, **x-handler** enables code↔API generation, **x-codeSamples** shows code on the right, `externalDocs` on the tag bridges to your concept pages. ([Redocly][3])

---

# H. Connect everything with CI

1. **Lint & bundle** OpenAPI:

```bash
redocly lint openapi/openapi.yaml && redocly bundle openapi/openapi.yaml -o openapi/bundled.yaml
```

Use `redocly.yaml` with the rules above (and any others you want from the recommended/minimal sets). ([Redocly][17])

2. **MkDocs build** runs the three generators:

* `gen_module_pages.py` (modules + nearest neighbors)
* `gen_api_tag_pages.py` (MkDocs pages per tag)
* `gen_api_diagrams.py` (API diagram with clickable nodes)

3. Optionally prebuild a **ReDoc static HTML** with options and ship it under `docs/api/reference.html`. ([GitHub][18])

---

## Quick checklist for your repo

* [ ] Root **tags** with **description**, **x-displayName**, and **externalDocs**
* [ ] **x-tagGroups** for 2-level nav grouping
* [ ] **operationId**: unique, URL-safe, stable (enforced by Redocly)
* [ ] **x-codeSamples** + realistic **examples** per operation
* [ ] **x-handler** (private) to bind operations to code symbols
* [ ] `gen_api_tag_pages.py` + `gen_api_diagrams.py` wired via `mkdocs-gen-files`
* [ ] Material **tags plugin** enabled; module pages populated with `api:<tag>` chips
* [ ] ReDoc configured (via plugin or prebuilt HTML) with sensible options
* [ ] CI: `redocly lint` + MkDocs `--strict` builds

If you want, share a small excerpt of your `openapi.yaml` (2–3 operations) and your package name; I’ll tailor the two generators to emit tag chips and cross-links for your exact repo layout, and propose a focused Redocly ruleset matching your naming conventions.

[1]: https://redocly.com/docs/redoc/v2.x/config "Configure Redoc"
[2]: https://squidfunk.github.io/mkdocs-material/plugins/tags/?utm_source=chatgpt.com "Built-in tags plugin - Material for MkDocs - GitHub Pages"
[3]: https://redocly.com/learn/openapi/openapi-visual-reference/operation?utm_source=chatgpt.com "Operation Object"
[4]: https://redocly.com/docs/cli/commands/lint?utm_source=chatgpt.com "lint"
[5]: https://github.com/blueswen/mkdocs-redoc-tag "GitHub - blueswen/mkdocs-redoc-tag: A MkDocs plugin supports adding Redoc to the page."
[6]: https://redocly.com/docs/cli/rules/oas/operation-singular-tag?utm_source=chatgpt.com "operation-singular-tag"
[7]: https://redocly.com/docs/realm/content/api-docs/openapi-extensions/x-tag-groups?utm_source=chatgpt.com "OpenAPI extension: x-tagGroups"
[8]: https://raw.githubusercontent.com/Redocly/redoc/master/docs/redoc-vendor-extensions.md "raw.githubusercontent.com"
[9]: https://redocly.com/docs/cli/rules/recommended?utm_source=chatgpt.com "Recommended ruleset"
[10]: https://redocly.com/docs-legacy/api-reference-docs/specification-extensions/x-code-samples?utm_source=chatgpt.com "x-codeSamples"
[11]: https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/?utm_source=chatgpt.com "Grouping Operations With Tags | Swagger Docs"
[12]: https://github.com/Redocly/redoc/issues/1625?utm_source=chatgpt.com "support operationRef or operationId references in the ..."
[13]: https://swagger.io/docs/specification/v3_0/openapi-extensions/?utm_source=chatgpt.com "OpenAPI Extensions | Swagger Docs"
[14]: https://fastapi.tiangolo.com/how-to/extending-openapi/?utm_source=chatgpt.com "Extending OpenAPI - FastAPI"
[15]: https://redocly.com/docs/cli/rules/oas/operation-operationId?utm_source=chatgpt.com "operation-operationId"
[16]: https://redocly.com/docs/redoc/v2.x/redoc-vendor-extensions?utm_source=chatgpt.com "Redoc vendor extensions"
[17]: https://redocly.com/docs/cli/guides/lint-and-bundle?utm_source=chatgpt.com "Lint and bundle API descriptions with Redocly CLI"
[18]: https://github.com/Redocly/redoc?utm_source=chatgpt.com "OpenAPI/Swagger-generated API Reference Documentation"



Heck yes — based on your repo, here’s a tight, repo-specific plan that (1) gives you **gold-class x-tag examples** you can paste into `openapi.yaml`, (2) shows **best-in-class tag chips** for a module page, and (3) proposes a **clean operations hierarchy** (tag groups) you can drive into diagrams and cross-links.

Your code already exposes a FastAPI app with three endpoints — `GET /healthz`, `POST /search`, and `POST /graph/concepts` — plus Problem Details handling and typed middleware/hooks we can reflect into OpenAPI via vendor extensions.   
You also ship the app under both `search_api.app` and the bridged `kgfoundry.search_api.app` namespace, which is perfect for a stable `x-handler` mapping.  
And you standardize error responses on **RFC 9457 Problem Details**. 

---

# 1) Two “gold-class” x-tag examples (drop-in OpenAPI)

These examples show: clean **tag taxonomy**, **x-tagGroups**, stable **operationId**, deep links back to code with **x-handler**, realistic **examples**/**x-codeSamples**, and a tag–to–docs bridge via `externalDocs`.

```yaml
openapi: 3.1.0
info:
  title: kgfoundry API
  version: 0.2.0
  description: Hybrid retrieval + graph concepts, with Problem Details on errors.
  x-logo: { url: /assets/logo.svg, altText: kgfoundry }

# Authoritative tags (pretty names + docs backlinks)
tags:
  - name: search
    x-displayName: Search & Retrieval
    description: Execute hybrid search (dense + sparse + KG boosts).
    externalDocs:
      description: Module page for search implementation
      url: /modules/search_api.app/#search        # MkDocs page you generate

  - name: graph
    x-displayName: Graph Concepts
    description: Concept discovery endpoints over the knowledge graph.
    externalDocs:
      description: Graph pipeline notes
      url: /architecture/graphs/

  - name: health
    x-displayName: Health & Readiness
    description: Liveness and readiness checks.

# Two-level nav in ReDoc
x-tagGroups:
  - name: Core
    tags: [search, graph]
  - name: Operations
    tags: [health]

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer

paths:
  /search:
    post:
      operationId: search.execute
      tags: [search]
      summary: Execute hybrid search (FAISS, BM25/SPLADE, KG boosts)
      description: >
        Runs multi-channel retrieval and fuses results via RRF, with knowledge-graph boosts.
        Returns a ranked list with per-signal scores. Errors are RFC 9457 Problem Details.
      security: [{ BearerAuth: [] }]
      x-handler: "search_api.app:search"           # exact handler symbol in your repo
      x-codeSamples:
        - lang: curl
          source: |
            curl -X POST https://api.example.com/search \
              -H "Authorization: Bearer $KEY" \
              -H "Content-Type: application/json" \
              -d '{"query":"vector store","k":5}'
        - lang: Python
          source: |
            from search_client import Client
            Client().search({"query": "vector store", "k": 5})
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: "#/components/schemas/SearchRequest" }
            examples:
              minimal: { value: { query: "faiss splade", k: 5 } }
      responses:
        "200":
          description: Ranked results
          content:
            application/json:
              schema: { $ref: "#/components/schemas/SearchResponse" }
        "4XX":
          description: Problem Details
          content:
            application/problem+json:
              schema: { $ref: "#/components/schemas/ProblemDetails" }

  /graph/concepts:
    post:
      operationId: graph.concepts.list
      tags: [graph]
      summary: List graph concepts matching a query
      description: >
        Returns concept candidates from the knowledge graph with simple matching and limits.
      security: [{ BearerAuth: [] }]
      x-handler: "search_api.app:graph_concepts"
      x-codeSamples:
        - lang: curl
          source: |
            curl -X POST https://api.example.com/graph/concepts \
              -H "Authorization: Bearer $KEY" \
              -H "Content-Type: application/json" \
              -d '{"q":"cancer","limit":10}'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                q: { type: string }
                limit: { type: integer, minimum: 1, default: 50 }
      responses:
        "200":
          description: Concept list
          content:
            application/json:
              schema:
                type: object
                properties:
                  concepts:
                    type: array
                    items:
                      type: object
                      properties:
                        concept_id: { type: string }
                        label: { type: string }
        "4XX":
          description: Problem Details
          content:
            application/problem+json:
              schema: { $ref: "#/components/schemas/ProblemDetails" }

  /healthz:
    get:
      operationId: infra.healthz
      tags: [health]
      summary: Health check
      description: Report component availability (bm25, splade, etc.)
      x-handler: "search_api.app:healthz"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema: { type: object, additionalProperties: true }

components:
  schemas:
    SearchRequest:
      type: object
      properties:
        query: { type: string }
        k: { type: integer, minimum: 1, default: 5 }
      required: [query, k]

    SearchResponse:
      type: object
      properties:
        results:
          type: array
          items:
            type: object
            properties:
              doc_id: { type: string }
              chunk_id: { type: string }
              title: { type: string }
              section: { type: string }
              score: { type: number }
              signals: { type: object, additionalProperties: { type: number } }
              spans: { type: object, additionalProperties: { type: number } }
              concepts:
                type: array
                items:
                  type: object
                  properties:
                    concept_id: { type: string }
                    label: { type: string }

    ProblemDetails:
      type: object
      description: RFC 9457 Problem Details
      properties:
        type: { type: string }
        title: { type: string }
        status: { type: integer }
        detail: { type: string }
        instance: { type: string }
```

Why these fit your repo:

* The `x-handler` values **exactly** match your handlers (`search_api.app:search`, `:graph_concepts`, `:healthz`) so your generators can auto-link code ↔ ReDoc.   
* Problem Details are first-class in your implementation; the spec declares a shared schema to make that visible. 
* Add (or enforce) stable `operationId`s — your site and diagrams will deep-link with `#operation/<operationId>`.
* `externalDocs` on tags points into your MkDocs pages (Architecture / Modules), so ReDoc is no longer an island.

> Tip: if you prefer to **generate** this file from FastAPI then post-process it, keep your Python routes but set `operation_id="search.execute"` on decorators, or run a tiny OpenAPI “augmenter” that merges the `tags`, `x-tagGroups`, and `x-handler` by operationId after export. (Happy to supply that script if you want it pre-wired.)

---

# 2) Best-in-class **tag chips** on a module page

Material’s tags + your generator can make modules feel “API-aware.” For the `search_api.app` module page:

```markdown
---
title: search_api.app
tags:
  # Product/domain
  - domain:retrieval
  - domain:graph
  # Layer / component
  - layer:interface
  - component:search_api
  # Ownership & lifecycle
  - owner:@search-api
  - status:experimental
  # API tags (bridge to ReDoc tag pages)
  - api:search
  - api:graph
---

# search_api.app

FastAPI endpoints for hybrid search and graph concepts.

## Related API operations

- **POST** `/search` — search.execute → [View in ReDoc](/api/index/#operation/search.execute)
- **POST** `/graph/concepts` — graph.concepts.list → [View in ReDoc](/api/index/#operation/graph.concepts.list)
- **GET** `/healthz` — infra.healthz → [View in ReDoc](/api/index/#operation/infra.healthz)
```

Where the chips come from:

* **Component / layer / owner / status** can be read from your module’s `__navmap__` (you already store `owner`, `stability`) and written into front-matter during generation. 
* **api:*** chips come from the OpenAPI tag(s) of operations whose `x-handler` points at this module.
* The links use the ReDoc deep-link (`#operation/<operationId>`) and create true two-way navigation.

---

# 3) Recommended **operations hierarchy** (tag groups) for diagrams & IA

Mirror your **package domains** + the operational reality of the system:

**Core**

* `search` – hybrid retrieval surface (your `/search` op). 
* `graph` – concept lookup (`/graph/concepts`). 

**Operations**

* `health` – liveness/readiness (`/healthz`). 

**Indexing & Data Plane** *(future-ready; reflects your codebase domains)*

* `ingest` – document/embedding ingestion (ties to `download`, `embeddings_dense`, `embeddings_sparse`). 
* `registry` – dataset/index metadata ops (ties to `registry/*`). 
* `kg` – graph build/queries beyond concepts (ties to `kg_builder/*`). 

**Observability & Admin**

* `observability` – metrics/logs/traces surfacing (you already centralize metrics and correlated logging in this service). 
* `config` – runtime/settings exposure (RuntimeSettings bootstrap). 

Use these as **`x-tagGroups`** in the spec (Core / Operations / Indexing & Data Plane / Observability & Admin). For diagrams, cluster per tag group and link every node to the ReDoc deep link.

**D2 diagram seed** (auto-generated from OpenAPI):

```d2
direction: right
APIs: "APIs" {
  Core: "Core" {
    "POST /search" { link: "../api/index/#operation/search.execute" }
    "POST /graph/concepts" { link: "../api/index/#operation/graph.concepts.list" }
  }
  Operations: "Operations" {
    "GET /healthz" { link: "../api/index/#operation/infra.healthz" }
  }
}
```

---

## Glue so it stays in sync (small generators)

1. **Augment OpenAPI** (merges tag groups + x-handler):

* Export the FastAPI schema (or bundle one you keep under `openapi/openapi.yaml`).
* Run a small Python script in your MkDocs `gen-files` step that:

  * injects the **root `tags`** and **`x-tagGroups`** blocks,
  * merges per-operation **`tags`** and **`x-handler`** by `operationId`,
  * writes `docs/api/openapi.yaml`.

2. **Per-tag MkDocs pages** (already discussed earlier) — list ops under each tag and deep-link into ReDoc.

3. **Module page chips** — during your one-module-one-page generation, add:

   * front-matter chips from `__navmap__` (`owner`, `stability`) and your architecture yml,
   * `api:*` chips from operations whose `x-handler` matches this module.

---

## Why this works for your tree (and scales)

* The **handlers and endpoints** are already cleanly separated (`search_api.app:search`, `:graph_concepts`, `:healthz`), so `x-handler` is stable and trivial to generate.   
* The app centralizes **metrics/logging/problem details**; surfacing those concerns via tags and trait pages keeps docs operationally useful.  
* The **namespaced bridge** `kgfoundry.search_api.app` lets you cross-link module pages under either import path cleanly. 

If you want, I can also drop in a tiny `augment_openapi.py` that reads your exported schema, merges the **`tags`/`x-tagGroups`**, attaches **`x-handler`** by `operationId`, and writes `docs/api/openapi.yaml` so your build is fully repeatable.


Great question — and I’m with you on **leaning on off-the-shelf tooling** wherever possible.

Short answer:

* Use **mkdocs-gen-files** to *generate* derived pages (tag pages, diagrams, per-module summaries).
* Use **Redocly CLI** to *augment/transform* the OpenAPI file (tags, x-tagGroups, x-handler, code samples) during a `bundle` step; Redocly is purpose-built for spec transforms via **decorators** and validation via **rules**. ([Oprypin][1])
* If you prefer “no transforms”, you can also **annotate your FastAPI routes directly** (set `operation_id`, `tags`, `openapi_extra` for vendor extensions like `x-handler`) so the exported spec already has the metadata. ([FastAPI][2])

Below are two “max-reuse” approaches (pick one), and then I include the Python fallback you asked for.

---

# Option A — Zero/low bespoke: put metadata in code + let Redocly bundle

## 1) Add the right metadata in FastAPI (no scripts)

FastAPI supports everything you need right on the decorators:

```python
from fastapi import FastAPI

app = FastAPI()

@app.post(
    "/search",
    operation_id="search.execute",          # stable deep-link anchor
    tags=["search"],                        # nav category in ReDoc
    openapi_extra={                         # vendor extensions
        "x-handler": "kgfoundry.search_api.app:search",
        "x-codeSamples": [
            {"lang": "curl", "source": "curl -X POST /search -d '{\"query\":\"kg\"}'"},
            {"lang": "Python", "source": "client.search({'query': 'kg'})"},
        ],
    },
)
async def search(...):
    ...
```

* `operation_id` gives you stable `#operation/<id>` deep links. ([FastAPI][2])
* `tags=[...]` puts the operation in a ReDoc tag section.
* `openapi_extra={...}` injects vendor extensions such as `x-handler` and `x-codeSamples`. ([Docs4Dev][3])

For tag cosmetics and nav:

* In the **root** of your OpenAPI, add `tags` with **`x-displayName`** (pretty tag labels) and **`x-traitTag`** (trait markers that don’t collect operations).
* Add **`x-tagGroups`** at the root to create a two-level nav; any tag not in a group is *not shown* in ReDoc, so list them all. ([Redocly][4])

You can keep those tag blocks in a tiny YAML file (`openapi/_topmatter.yaml`) and **concatenate** them into your exported spec with Redocly (next step), so your app code stays minimal.

## 2) Bundle, validate, and emit the doc for MkDocs (all tooling)

```yaml
# redocly.yaml
apis:
  main:
    root: openapi/base.yaml               # or point to your FastAPI-exported spec
    decorators:
      remove-unused-components: on        # tidies the bundled file
    rules:
      operation-operationId: error
      operation-operationId-unique: error
      # Example of a configurable rule (casing, etc.)
      rule/operationId-casing:
        subject: { type: Operation, property: operationId }
        assertions: { casing: camelCase }
```

```bash
# Validate and produce the single file your MkDocs page will load
redocly lint openapi/base.yaml
redocly bundle openapi/base.yaml -o docs/api/openapi.yaml
```

* **Decorators** are Redocly’s official way to add/transform content during bundling; you can do a lot without writing custom code.
* **Rules** enforce hygiene (e.g., “every operation must have a unique, URL-safe `operationId`”). ([Redocly][5])

> Why this fits your “use libraries” goal: the only code you wrote is in your FastAPI route decorators; everything else is handled by Redocly’s built-in machinery.

---

# Option B — Still minimal bespoke, but *data-driven* augmentation (Redocly plugin)

If you can’t (or don’t want to) touch the FastAPI code, keep a **data file** that maps `operationId` → tags / x-handler / code samples, and let Redocly add it during `bundle`. Redocly supports **custom decorators** (plugins) for exactly this kind of transform. ([Redocly][6])

### 1) A tiny data file you maintain

```yaml
# openapi/_augment.yaml
tags:
  - name: search
    x-displayName: Search & Retrieval
    description: Execute hybrid search.
  - name: graph
    x-displayName: Graph Concepts
    description: Concept discovery.
x-tagGroups:
  - name: Core
    tags: [search, graph]

operations:
  search.execute:
    tags: [search]
    x-handler: "kgfoundry.search_api.app:search"
    x-codeSamples:
      - {lang: curl, source: "curl -X POST /search -d '{\"query\":\"kg\"}'"}
  graph.concepts.list:
    tags: [graph]
    x-handler: "kgfoundry.search_api.app:graph_concepts"
```

### 2) 30-line Redocly decorator (generic)

```js
// plugins/inject-metadata.js
import fs from 'node:fs';
import yaml from 'js-yaml';

export default function injectMetadata() {
  const aug = yaml.load(fs.readFileSync('openapi/_augment.yaml', 'utf8')) || {};
  const ops = aug.operations || {};
  const rootTags = aug.tags || [];
  const tagGroups = aug['x-tagGroups'] || [];

  return {
    id: 'inject-metadata',
    decorators: {
      oas3: {
        Root: {
          leave(root) {
            root.tags = [...(root.tags || []), ...rootTags];
            if (tagGroups.length) root['x-tagGroups'] = tagGroups;
          },
        },
        Operation: {
          leave(op) {
            const id = op.operationId;
            if (!id || !ops[id]) return;
            const add = ops[id];
            // merge tags (set union)
            op.tags = Array.from(new Set([...(op.tags || []), ...(add.tags || [])]));
            // copy selected vendor extensions
            for (const k of Object.keys(add)) {
              if (k.startsWith('x-')) op[k] = add[k];
            }
          },
        },
      },
    },
  };
}
```

### 3) Redocly config to enable it

```yaml
# redocly.yaml
apis:
  main:
    root: openapi/base.yaml
plugins:
  - ./plugins/inject-metadata.js
decorators:
  inject-metadata/Root: on
  inject-metadata/Operation: on
rules:
  operation-operationId: error
  operation-operationId-unique: error
```

```bash
redocly lint openapi/base.yaml
redocly bundle openapi/base.yaml -o docs/api/openapi.yaml
```

This keeps “code” to a single generic plugin and a data file; everything else is Redocly’s engine (lint, bundle, rules). ([Redocly][5])

---

# Where **mkdocs-gen-files** fits (and why not to mutate the spec there)

* **Use it for derived docs**: generating *per-tag* MkDocs pages, a ReDoc-backed *API landing page*, and *D2 diagrams* that deep-link to `#operation/<operationId>`. That’s its sweet spot: writing virtual files during the MkDocs build, with edit links mapped to your code. ([Oprypin][1])
* **Avoid using it to rewrite the spec**: spec transforms belong in the **API toolchain** (lint → decorate → bundle) so you can validate, test, and reuse the same single `docs/api/openapi.yaml` in other contexts (SDK gen, portals). Redocly decorators exist for exactly that, and they run *during bundling*. ([Redocly][7])

If you still want a Python augmenter, see the fallback below (it plugs into `gen-files` cleanly).

---

# (Fallback) Python `augment_openapi.py` you asked for

If you’d like everything in Python, this drops into `plugins.gen-files.scripts` and writes the final spec to `docs/api/openapi.yaml`. It merges a base spec with `_augment.yaml` (same structure as above) by **operationId** and appends root tags and `x-tagGroups`.

```python
# docs/_scripts/augment_openapi.py
from __future__ import annotations
import yaml, mkdocs_gen_files
from pathlib import Path

BASE = Path("openapi/base.yaml")          # your exported or hand-written spec
AUG  = Path("openapi/_augment.yaml")      # tags, x-tagGroups, operations: {id: {...}}

def load(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

base = load(BASE)
aug  = load(AUG)

# 1) Append/merge root tags and x-tagGroups
base.setdefault("tags", [])
if aug.get("tags"):
    # de-duplicate by tag 'name'
    existing = {t["name"] if isinstance(t, dict) else t for t in base["tags"]}
    for t in aug["tags"]:
        name = t["name"] if isinstance(t, dict) else t
        if name not in existing:
            base["tags"].append(t)

if aug.get("x-tagGroups"):
    base["x-tagGroups"] = aug["x-tagGroups"]

# 2) Merge per-operation metadata by operationId
op_meta: dict = (aug.get("operations") or {})
for path, item in (base.get("paths") or {}).items():
    for method, op in list(item.items()):
        if method.upper() not in {"GET","PUT","POST","DELETE","PATCH","HEAD","OPTIONS","TRACE"}:
            continue
        opid = op.get("operationId")
        if not opid or opid not in op_meta:
            continue
        add = op_meta[opid]
        # tags: union
        if "tags" in add:
            op["tags"] = sorted(set((op.get("tags") or []) + add["tags"]))
        # vendor extensions: copy keys that start with x-
        for k, v in add.items():
            if k.startswith("x-"):
                op[k] = v

# 3) Write output into the virtual docs dir
with mkdocs_gen_files.open("api/openapi.yaml", "w") as f:
    yaml.safe_dump(base, f, sort_keys=False)
```

Then wire it up in `mkdocs.yml`:

```yaml
plugins:
  - gen-files:
      scripts:
        - docs/_scripts/augment_openapi.py
```

This works fine, but you lose **linting/rules** unless you also run Redocly or Spectral separately. Redocly’s “rules” and “decorators” are the bigger win for long-term maintenance. ([Redocly][8])

---

## Bonus: make tags shine across your site

* **Pretty tags & groups** in ReDoc: use `x-displayName`, `x-traitTag`, `x-tagGroups`. ([Redocly][4])
* **Show schemas by tag** with `x-tags` on component schemas, so ReDoc lists domain models next to their operations. ([Redocly][9])
* **Deep links everywhere**: from module pages, D2 diagrams, and per-tag MkDocs pages jump to `…/api/index/#operation/<operationId>`. ReDoc treats `operationId` as the deep-link fragment. ([Redocly][10])

---

## Recommendation (TL;DR)

* Prefer **Option A** (metadata in FastAPI + Redocly bundle/lint). It’s the least custom, uses vendor-supported features end-to-end, and keeps your spec authoritative. ([FastAPI][2])
* Use **mkdocs-gen-files** for *pages/diagrams*, *not spec mutation*. ([Oprypin][1])
* Keep **Option B** (Redocly plugin + YAML data) if you can’t change route decorators.
* The **Python augmenter** above is there if you want everything in-repo and in-Python, but I’d still run Redocly `lint` in CI for guardrails. ([Redocly][8])

If you tell me which path you prefer (A, B, or Python fallback), I’ll tailor the files to your exact package/module names and drop in the matching MkDocs nav + build commands.

[1]: https://oprypin.github.io/mkdocs-gen-files/api.html?utm_source=chatgpt.com "API - mkdocs-gen-files"
[2]: https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/?utm_source=chatgpt.com "Path Operation Advanced Configuration - FastAPI"
[3]: https://www.docs4dev.com/docs/fastapi/latest/advanced/path-operation-advanced-configuration/?utm_source=chatgpt.com "FastAPI"
[4]: https://redocly.com/docs/api-reference-docs/specification-extensions/x-display-name?utm_source=chatgpt.com "x-displayName"
[5]: https://redocly.com/docs/cli/configuration/reference/decorators?utm_source=chatgpt.com "decorators"
[6]: https://redocly.com/docs/cli/custom-plugins/custom-decorators/?utm_source=chatgpt.com "Decorators in plugins"
[7]: https://redocly.com/docs/cli/decorators?utm_source=chatgpt.com "Decorators"
[8]: https://redocly.com/docs/cli/v1/configuration/reference/rules?utm_source=chatgpt.com "rules"
[9]: https://redocly.com/docs/realm/content/api-docs/openapi-extensions/x-tags?utm_source=chatgpt.com "OpenAPI extension: x-tags"
[10]: https://redocly.com/docs/openapi-visual-reference/operation/?utm_source=chatgpt.com "Operation Object"
