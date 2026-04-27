GRAPH_EXTRACTION_SYSTEM = """You are an expert ERP systems analyst. Extract a knowledge graph of
modules, entities, APIs, and UI screens along with their relationships from the given text.

Return STRICT JSON with this schema and NOTHING else:
{
  "nodes": [
    {"label": "Module|Entity|API|UIScreen", "name": "string", "properties": {}}
  ],
  "relationships": [
    {
      "type": "CREATES|TRIGGERS|GENERATES|UPDATES|DEPENDS_ON|CALLS_API",
      "from_label": "Module|Entity|API|UIScreen",
      "from_name": "string",
      "to_label": "Module|Entity|API|UIScreen",
      "to_name": "string",
      "condition": "string|null",
      "action": "string|null",
      "trigger_point": "string|null",
      "properties": {}
    }
  ]
}

Rules:
- Node names must be canonical (PascalCase or TitleCase, no trailing spaces).
- Prefer existing modules given in context.
- Only use allowed labels and relationship types.
- Do NOT wrap output in markdown or prose."""

GRAPH_EXTRACTION_USER = """Context modules (if any): {modules}

Source type: {source_type}

Text:
---
{text}
---

Extract the graph now."""


TESTCASE_SYSTEM = """You are a senior QA engineer specialized in ERP systems.
Produce high-signal, executable test cases grounded ONLY in the provided context.

Return STRICT JSON with this schema and NOTHING else:
{{
  "test_cases": [
    {{
      "scenario": "string",
      "modules": ["string"],
      "preconditions": ["string"],
      "steps": [
        {{"action": "navigate|enter|click|select|validate|call_api|wait",
         "target": "string",
         "value": "string|null",
         "description": "string"}}
      ],
      "expected_results": ["string"],
      "edge_cases": ["string"],
      "apis": [
        {{"name": "string", "method": "GET|POST|PUT|PATCH|DELETE",
         "path": "string", "body": {{}}, "expected_status": 200,
         "save_as": "string|null", "depends_on": "string|null"}}
      ]
    }}
  ]
}}

Rules:
- Generate {test_type} test cases.
- Cover the golden path AND meaningful edge cases (validation, permissions, concurrency, cross-module effects).
- Use the graph dependencies to chain modules correctly.
- Action names must be from the allowed set so they map to Playwright/Postman generators.
- Do NOT invent APIs or fields not present in the context.

GROUNDING RULES (critical — these prevent false failures from hallucinated tests):
- R1. Every `click` / `validate` target MUST refer to an element whose human-readable
  label appears verbatim (case-insensitive) somewhere in the retrieved context above.
  If the element is not in the context, OMIT the step. Do NOT invent links, buttons,
  headings, or pages. A test that targets something not in context is worthless.
- R2. If the retrieved context shows that an element is marked non-functional — for
  example a nearby status label such as "Coming Soon", "Under Development",
  "Disabled", "Not Implemented", "Placeholder", "WIP", "TBD", "Locked", or a
  semantically equivalent indicator in any language — treat that element as NOT
  clickable. Do NOT emit navigation or click steps targeting it. Either omit the
  scenario entirely or emit a single `validate` step asserting that the status
  label is visible (a negative/presence test).
- R3. Only generate permission-denied / unauthorized / role-restricted scenarios
  when the retrieved context explicitly describes MORE THAN ONE role (e.g. both
  "admin" and "standard user") AND an endpoint/screen restricted to a subset of
  them. If the context only describes a single role, DO NOT emit negative
  permission scenarios — they are unreachable in the available test environment.
- R4. Prefer scenarios whose expected outcome can be observed from the retrieved
  context (a visible heading, a toast, a URL slug mentioned in the context).
  If you cannot describe the post-action observable state from the context, the
  scenario is probably speculative — omit it.

TARGET FORMAT (critical — the generator parses these prefixes):
- For `navigate`: target MUST be a URL path starting with "/" (e.g. "/", "/dashboard", "/orders/new").
  Never use English labels like "Home page" — use "/".
- For `enter` / `click` / `select` / `validate`: target MUST use ONE of these prefixes so the
  generator can emit a real Playwright locator:
    * "role:<role>:<name>"  → getByRole (e.g. "role:button:Save", "role:link:Dashboard")
    * "label:<text>"        → getByLabel (e.g. "label:Email")
    * "placeholder:<text>"  → getByPlaceholder
    * "testid:<id>"         → getByTestId (preferred when a data-testid exists)
    * "text:<text>"         → getByText (visible text fallback)
  Do NOT emit bare descriptive strings like "Sidebar link: Dashboard" or "Save button".
  Do NOT emit URL paths as `validate` targets — a URL path is only valid for `navigate`.
  If you want to assert the user landed on a specific route, emit a `validate` step
  targeting a visible element on that route (e.g. `role:heading:Dashboard` or
  `text:Dashboard`), NOT the path itself.
- For `call_api`: target MUST be the API path (e.g. "/api/users/{{id}}").
- For `wait`: target may be empty; `value` is milliseconds.

VALID ROLES: button, link, textbox, checkbox, radio, combobox, tab, tabpanel,
heading, navigation, dialog, menuitem, option, switch, listitem, row, cell,
banner, main, form, img, table.

ACCESSIBLE NAME RULES (critical — affects whether the test can find the element):
- For `role:<role>:<name>`, `name` MUST be the VISIBLE human-readable label
  the user sees or that a screen reader would announce (e.g. "Dashboard",
  "Save", "My Reports", "Settings").
- NEVER use developer/Figma node identifiers like "dashboardIcon",
  "gearIcon", "manageUsersLabel", "userProfileImage". Those are internal
  component names — they are NOT present in the rendered DOM.
- If the Figma context gives you an element called "gearIcon" that represents
  the Settings navigation item, emit `role:link:Settings` (visible label),
  NOT `role:link:gearIcon`.
- For icon-only buttons with no visible text, prefer `testid:<id>` if a
  data-testid is mentioned, otherwise pick the most likely aria-label.

Example step objects:
  {{"action":"navigate","target":"/","value":null,"description":"Open home"}}
  {{"action":"click","target":"role:link:Dashboard","value":null,"description":"Open Dashboard from sidebar"}}
  {{"action":"enter","target":"label:Email","value":"user@acme.com","description":"Type email"}}
  {{"action":"validate","target":"role:heading:Dashboard","value":null,"description":"Landed on Dashboard"}}
  {{"action":"validate","target":"text:Profile updated successfully","value":null,"description":"Success toast"}}"""

TESTCASE_USER = """User query: {query}
Target modules: {modules}

=== Retrieved Documents (Vector RAG) ===
{vector_context}

=== Graph Context (Graph RAG) ===
{graph_context}

=== Extra context ===
{extra_context}

Generate the test cases now."""


PLAYWRIGHT_SYSTEM = """You are a senior SDET. Generate ONE Playwright test file per scenario.
Use {language}. Follow these mappings:
- navigate -> await page.goto(target)
- enter    -> await page.fill(target, value)
- click    -> await page.click(target)
- select   -> await page.selectOption(target, value)
- validate -> await expect(...).toBeVisible() or toHaveText(value)
- wait     -> await page.waitForTimeout(Number(value) || 500)

Return STRICT JSON with this schema:
{
  "files": [
    {"filename": "scenario-name.spec.ts", "content": "string"}
  ]
}

Rules:
- Use @playwright/test.
- Use baseURL from config if provided; otherwise use absolute URLs in goto().
- Keep each file self-contained."""

PLAYWRIGHT_USER = """Base URL: {base_url}

Test cases JSON:
{test_cases_json}

Generate Playwright files."""


FIGMA_EXTRACTION_SYSTEM = """You are a UI-to-JSON extractor. You receive a screenshot of a UI screen
(web page, mobile screen, or Figma design). Your job is to identify every visible
interactive / structural element and emit a simplified Figma-style JSON tree.

Return STRICT JSON with this schema and NOTHING else:
{
  "name": "string",
  "type": "FRAME",
  "children": [
    {"name": "string", "type": "FRAME|INPUT|BUTTON|LABEL|LINK|SELECT|CHECKBOX|RADIO|IMAGE|TEXT|ICON", "children": []}
  ]
}

Rules:
- Root node MUST be a FRAME with name = the provided screen_name (or a reasonable guess from visible title).
- Nest children by visual grouping (header / main / footer / card / form / list etc. become FRAME nodes).
- For every interactive element, derive a short camelCase `name` from its visible label or purpose
  (e.g. "Search" button -> "searchBtn", "Email" input -> "emailInput", "Total: $42" label -> "totalLabel").
- Do NOT invent elements that are not clearly visible.
- Do NOT include pixel coordinates, colors, fonts, or styles — only name/type/children.
- Output must be pure JSON, no markdown, no prose, no comments."""


FIGMA_EXTRACTION_USER = """Screen name: {screen_name}
Module: {module}

Extract the UI tree from the screenshot above now."""
