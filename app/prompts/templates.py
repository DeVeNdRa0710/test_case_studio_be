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
- R5. NEVER emit login/sign-in steps. The Playwright harness logs in
  automatically via a `login(page)` helper in `beforeEach` whenever a scenario
  needs an authenticated user. Do NOT emit any of: `navigate /login`,
  `enter Username|Email|Employee ID`, `enter Password`, or `click Sign in|Login`.
  Express the auth requirement once, in `preconditions`, with a phrase like
  "User is logged in as <role>". The first step of an authenticated scenario
  should be the FIRST POST-LOGIN ACTION (e.g. navigate to /dashboard, click a
  nav link), never the login itself.
- R6. NEVER invent record identifiers, primary keys, or seed data the context
  did not give you. Do NOT emit targets like `text:Pending RCA Report:
  Equipment Malfunction` or routes like `/rca/RCA-001/edit` unless that exact
  record name or ID appears in the retrieved context. If the scenario needs
  a specific record on a list page, you MUST use one of these structural
  sentinels (the test runtime recognizes them and clicks the first real list
  row, table row, or card):
    * `text:first row`
    * `text:first record`
    * `text:any pending row`
    * `text:any <status> <noun> row`   (e.g. `text:any pending FAR row`)
    * `text:the first item`
  The sentinel MUST start with `any`, `first`, `a`, `the first`, or `the
  next`, and MUST end with the singular or plural form of `row`, `record`,
  `entry`, `item`, `card`, `line`, `listing`, or `result`. Anything outside
  this shape is treated as literal text and will fail to find a record.
  A test that depends on hand-invented record IDs will always fail in the
  real environment.
- R7. NEVER emit a `navigate` to a URL path that is not present in the
  retrieved context. Routes like `/approvals`, `/horizontal-deployment`,
  `/rca/<id>/edit` are guesses unless the context shows them. If the context
  only proves a screen exists by name (not by route), emit a `click` on the
  visible link/menu item that opens it instead of guessing the path.
- R8. The "Extra context" block (when non-empty) describes the LIVE,
  IMPLEMENTED application — what users actually see in the deployed UI today.
  It OVERRIDES Figma/requirements vocabulary whenever the two conflict.
  Specifically:
    a. If "Extra context" lists a sidebar/nav menu, ONLY use those labels
       in `click` targets for navigation — never invent labels that aren't
       in that list, even if a Figma frame uses different terminology.
    b. If "Extra context" marks a screen as "Coming Soon" / "Not Implemented" /
       "Disabled" / "WIP", treat that screen as unreachable: do NOT emit
       click/navigate steps that depend on it; either omit the scenario or
       degrade it to a `validate` step asserting the "Coming Soon" label is
       visible.
    c. If "Extra context" gives field labels for a form (e.g. the actual
       inputs are "What?", "Where?", "When?", "Loss Impact"), use THOSE
       in `enter` / `select` targets — never the Figma/spec labels like
       "RCA Description" or "Root Cause" that aren't in the live form.
    d. If "Extra context" gives a route/sitemap, use THOSE paths in
       `navigate` targets instead of guessing.
  When Figma and Extra context disagree, Extra context wins. Figma describes
  intent; Extra context describes reality.
- R9. SHALLOW SCENARIOS BY DEFAULT. Generate only as many steps as the
  Extra context PROVES are real. After each step, ask yourself: "does the
  Extra context describe what the user sees on the screen this step would
  produce, including the next field/button I'm about to interact with?"
  If the answer is no, STOP the scenario. Emit a `validate` step that
  asserts the last verified element is visible, then end.
  Concretely:
    a. If Extra context lists a list page (e.g. "My Reports") but does NOT
       enumerate the detail page that opens when you click a row, your
       scenario must end with `validate` on the row, not click further.
    b. If Extra context lists form fields A, B, C but does NOT mention a
       Submit button, end with the last `enter`. Do NOT invent a Submit.
    c. If Extra context lists a tab (e.g. "Action Plan" inside FIR) but
       does NOT enumerate that tab's fields, do NOT generate steps for
       fields inside that tab. A 3-step scenario that passes is infinitely
       more useful than a 12-step scenario that dies at step 4.
  A test that ends early with `validate` is a SMOKE TEST — it proves the
  entry point is reachable. That is a complete, valuable test. Do not
  apologize for stopping early; stopping early is the correct behavior
  when context runs out.
- R10. NEVER fabricate the post-row-click detail page. List/table pages
  in real apps render either (a) inline expand/collapse, (b) a side
  drawer, or (c) a route change. You CANNOT predict which without the
  Extra context telling you. So after a `text:any <X> row` click, your
  ONLY allowed next steps are:
    * `validate` something the Extra context says appears post-click, OR
    * end the scenario.
  Do NOT emit a `click` on "Approve"/"Submit"/"Save"/"Edit" or any other
  action button after a row click unless that exact button label is named
  in the Extra context as appearing on the post-click detail view.

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
