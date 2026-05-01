"""
Deterministic Playwright spec emitter.

Takes structured test cases from the LLM and emits:
  - one helpers/auth.ts file (login helper + BASE_URL)
  - one <scenario>.spec.ts file per test case

Target format produced by the LLM (enforced via TESTCASE_SYSTEM prompt):
  - navigate: target = URL path ("/", "/dashboard")
  - click/enter/select/validate: target = one of
      "role:<role>:<name>" | "label:<text>" | "placeholder:<text>"
      "testid:<id>"        | "text:<text>"
  - call_api: target = API path
"""
from __future__ import annotations

import json as _json
import re
from typing import Any

from app.schemas.testcase import TestCase


# ---------------- helpers ----------------

def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "scenario"


def _esc_ts(value: Any) -> str:
    """Escape a value for safe interpolation inside single-quoted TS strings."""
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _esc_regex(value: Any) -> str:
    """Escape a value for safe use inside a TS regex literal."""
    if value is None:
        return ""
    return re.sub(r"[.*+?^${}()|[\]\\/]", r"\\\g<0>", str(value))


def _needs_login(test_case: TestCase) -> bool:
    text = " ".join(test_case.preconditions or []).lower()
    if any(k in text for k in ("logged in", "login", "authenticated", "signed in")):
        return True
    return _has_login_shaped_steps(test_case.steps or [])


_LOGIN_USER_LABEL_RE = re.compile(
    r"(user.?name|email|user.?id|employee.?id|employee.?code|emp.?id|emp.?code|login|user.?code)",
    re.IGNORECASE,
)
_LOGIN_PASS_LABEL_RE = re.compile(r"(password|passcode|\bpass\b|\bpin\b)", re.IGNORECASE)
_LOGIN_SUBMIT_LABEL_RE = re.compile(
    r"(log ?in|sign ?in|submit|continue)", re.IGNORECASE
)


def _looks_like_login_path(target: str) -> bool:
    t = (target or "").strip().lower()
    return t in {"/login", "/signin", "/sign-in", "/log-in", "/auth/login", "/auth/signin"}


def _step_target_label(step: dict[str, Any]) -> str:
    """Strip role:/label:/etc. prefix and return the human label, lowercased."""
    return _strip_prefix(step.get("target") or "").lower()


def _is_login_step(step: dict[str, Any]) -> bool:
    """Heuristic: does this step look like part of a login form interaction?"""
    action = (step.get("action") or "").lower()
    if action == "navigate" and _looks_like_login_path(step.get("target") or ""):
        return True
    if action == "enter":
        label = _step_target_label(step)
        if _LOGIN_USER_LABEL_RE.search(label) or _LOGIN_PASS_LABEL_RE.search(label):
            return True
    if action == "click":
        label = _step_target_label(step)
        if _LOGIN_SUBMIT_LABEL_RE.search(label):
            return True
    return False


def _has_login_shaped_steps(steps: list[dict[str, Any]]) -> bool:
    """True if a head-of-list run of steps looks like a manual login sequence
    (a username-ish enter and a password-ish enter, optionally bracketed by
    /login navigate and a sign-in click)."""
    saw_user = False
    saw_pass = False
    for step in steps[:6]:
        if not _is_login_step(step):
            break
        action = (step.get("action") or "").lower()
        if action == "enter":
            label = _step_target_label(step)
            if _LOGIN_USER_LABEL_RE.search(label):
                saw_user = True
            elif _LOGIN_PASS_LABEL_RE.search(label):
                saw_pass = True
    return saw_user and saw_pass


def _strip_leading_login_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the head-of-list login sequence — the `login(page)` helper in
    auth.ts handles auth deterministically, so inline login steps are at best
    redundant and at worst (when the LLM invents field labels like 'Username'
    that don't match the real form's 'Employee ID') a guaranteed failure."""
    if not _has_login_shaped_steps(steps):
        return list(steps)
    out = list(steps)
    while out and _is_login_step(out[0]):
        out.pop(0)
    return out


# ---------------- target parsing ----------------

_PREFIXES = ("role:", "label:", "placeholder:", "testid:", "text:")


_DEV_SUFFIXES = (
    "Icon", "Label", "Btn", "Button", "Image", "Img", "Link",
    "Field", "Input", "Text", "Node", "Element",
)


def _normalize_accessible_name(name: str) -> str:
    """Turn a dev-ish identifier like 'dashboardIcon' or 'manageUsersLabel'
    into what the DOM's accessible name is likely to be (e.g. 'dashboard',
    'manage users'). Strips a trailing UI-type suffix and splits camelCase."""
    out = name.strip()
    for suf in _DEV_SUFFIXES:
        if out.endswith(suf) and len(out) > len(suf):
            out = out[: -len(suf)]
            break
    # Split camelCase/PascalCase into space-separated words
    out = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", out)
    out = re.sub(r"[_-]+", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out or name.strip()


def _locator(target: str) -> str:
    """Convert a structured target string into a Playwright locator expression.

    Every locator includes a getByText fallback via `.or(...)` so tests also
    succeed on SPAs that use <div>/<span> with click handlers instead of
    semantic <a>/<button> elements.
    """
    t = (target or "").strip()
    if not t:
        return "page.locator('body').first()"

    if t.startswith("role:"):
        rest = t[len("role:"):].strip()
        if ":" in rest:
            role, name = rest.split(":", 1)
            normalized = _normalize_accessible_name(name.strip())
            pattern = f"/{_esc_regex(normalized)}/i"
            role_clean = _esc_ts(role.strip())
            # Fall back to visible-only text match so we skip hidden section
            # labels (e.g. Tailwind `hidden` class) that would otherwise win.
            return (
                f"page.getByRole('{role_clean}', {{ name: {pattern} }})"
                f".or(page.getByText({pattern}).and(page.locator(':visible')))"
                f".first()"
            )
        return f"page.getByRole('{_esc_ts(rest)}').first()"

    if t.startswith("label:"):
        pattern = f"/{_esc_regex(t[len('label:'):])}/i"
        return (
            f"page.getByLabel({pattern})"
            f".or(page.getByPlaceholder({pattern}))"
            f".first()"
        )

    if t.startswith("placeholder:"):
        return f"page.getByPlaceholder(/{_esc_regex(t[len('placeholder:'):])}/i).first()"

    if t.startswith("testid:"):
        return f"page.getByTestId('{_esc_ts(t[len('testid:'):])}')"

    if t.startswith("text:"):
        return (
            f"page.getByText(/{_esc_regex(t[len('text:'):])}/i)"
            f".and(page.locator(':visible')).first()"
        )

    # Fallback: treat as fuzzy visible text
    return (
        f"page.getByText(/{_esc_regex(t)}/i)"
        f".and(page.locator(':visible')).first()"
    )


def _url_path(target: str) -> str:
    """Sanitize a navigate target into a URL path."""
    t = (target or "").strip()
    if not t:
        return "/"
    if t.startswith("/") or t.startswith("http://") or t.startswith("https://"):
        return t
    # LLM gave us a descriptive label — fall back to "/"
    return "/"


# ---------------- step → code ----------------

def _strip_prefix(target: str) -> str:
    """Strip role:/label:/etc. prefix, return just the human label."""
    t = (target or "").strip()
    for prefix in ("role:", "label:", "placeholder:", "testid:", "text:"):
        if t.startswith(prefix):
            rest = t[len(prefix):]
            if prefix == "role:" and ":" in rest:
                rest = rest.split(":", 1)[1]
            return _normalize_accessible_name(rest)
    return _normalize_accessible_name(t)


def _region_hint(target: str) -> str | None:
    """If the target hints at a region (nav / header / main), return it."""
    t = (target or "").lower()
    if "nav" in t or "sidebar" in t or "menu" in t:
        return "nav"
    if "header" in t or "topbar" in t:
        return "header"
    if "dialog" in t or "modal" in t:
        return "dialog"
    return None


def _step_to_line(step: dict[str, Any]) -> str:
    action = (step.get("action") or "").lower()
    target = step.get("target") or ""
    value = step.get("value")
    description = step.get("description") or ""

    if action == "navigate":
        return f"  await t.goto(page, '{_esc_ts(_url_path(target))}');"

    if action == "enter":
        label = _strip_prefix(target)
        return f"  await t.fill(page, '{_esc_ts(label)}', '{_esc_ts(value)}');"

    if action == "click":
        label = _strip_prefix(target)
        region = _region_hint(description) or _region_hint(target)
        opts = f", {{ in: '{region}' }}" if region else ""
        return f"  await t.click(page, '{_esc_ts(label)}'{opts});"

    if action == "select":
        label = _strip_prefix(target)
        return f"  await t.select(page, '{_esc_ts(label)}', '{_esc_ts(value)}');"

    if action == "validate":
        label = _strip_prefix(target)
        if value is not None and str(value).strip():
            return f"  await t.see(page, '{_esc_ts(value)}');"
        # Defensive guard: a validate target must be a human-readable label, not a
        # URL path. If the LLM slipped a path in (e.g. "/dashboard") it would
        # produce an assertion that can never match visible text. Skip such steps
        # with an inline comment rather than emit broken code.
        has_prefix = any(
            (target or "").strip().startswith(p) for p in _PREFIXES
        )
        if not has_prefix and label.startswith("/"):
            return (
                f"  // skipped validate: target '{_esc_ts(target)}' looks like a URL path, "
                f"not a visible label. Use role:/text:/heading: prefix instead."
            )
        return f"  await t.see(page, '{_esc_ts(label)}');"

    if action == "wait":
        try:
            ms = int(value) if value is not None else 500
        except (TypeError, ValueError):
            ms = 500
        return f"  await page.waitForTimeout({max(0, ms)});"

    if action == "call_api":
        method = (step.get("method") or "GET").lower()
        path = _esc_ts(target)
        body = step.get("body")
        if body:
            body_literal = _json.dumps(body)
            return (
                f"  await request.{method}('{path}', "
                f"{{ data: {body_literal} }});"
            )
        return f"  await request.{method}('{path}');"

    desc = _esc_ts(description or action or "step")
    return f"  // {desc}"


# ---------------- file rendering ----------------

def _auth_helper_file(base_url: str | None) -> dict[str, str]:
    base = _esc_ts(base_url or "http://localhost:3000")
    content = f"""const {{ expect }} = require('@playwright/test');

const BASE_URL = process.env.BASE_URL ?? '{base}';
const LOGIN_URL = process.env.LOGIN_URL ?? '/login';

/**
 * Robust login helper. Tries several field-matching strategies so it works
 * across apps where the login form might use "Email", "Username", "Employee
 * ID", "User Code", etc. — with or without <label> elements.
 *
 * Override defaults via env vars if any heuristic misses:
 *   LOGIN_URL            (default: /login)
 *   LOGIN_USERNAME_CSS   (e.g. '#empId')
 *   LOGIN_PASSWORD_CSS   (e.g. '#password')
 *   LOGIN_SUBMIT_CSS     (e.g. 'button.signin')
 */
async function login(page) {{
  const username = process.env.APP_USERNAME;
  const password = process.env.APP_PASSWORD;
  if (!username || !password) {{
    throw new Error('APP_USERNAME and APP_PASSWORD must be set in the environment.');
  }}

  await page.goto(LOGIN_URL);
  await page.waitForLoadState('domcontentloaded');

  const userPattern = /email|user.?name|user.?id|employee.?id|employee.?code|emp.?id|emp.?code|login|code/i;

  await fillFirst(
    page,
    'username/email/employee-id',
    [
      process.env.LOGIN_USERNAME_CSS
        ? () => page.locator(process.env.LOGIN_USERNAME_CSS)
        : null,
      () => page.getByLabel(userPattern),
      () => page.getByPlaceholder(userPattern),
      () =>
        page.locator(
          'input[name*="user" i], input[name*="email" i], input[name*="login" i], ' +
          'input[name*="emp" i], input[name*="code" i], ' +
          'input[id*="user" i], input[id*="email" i], input[id*="login" i], ' +
          'input[id*="emp" i], input[id*="code" i]',
        ),
      () =>
        page
          .locator('input:not([type="password"]):not([type="hidden"]):not([type="submit"])')
          .first(),
    ],
    username,
  );

  await fillFirst(
    page,
    'password',
    [
      process.env.LOGIN_PASSWORD_CSS
        ? () => page.locator(process.env.LOGIN_PASSWORD_CSS)
        : null,
      () => page.getByLabel(/password|pass|pin/i),
      () => page.getByPlaceholder(/password|pass|pin/i),
      () => page.locator('input[type="password"]'),
    ],
    password,
  );

  await clickFirst(
    page,
    'submit',
    [
      process.env.LOGIN_SUBMIT_CSS
        ? () => page.locator(process.env.LOGIN_SUBMIT_CSS)
        : null,
      () => page.getByRole('button', {{ name: /log ?in|sign ?in|submit|continue/i }}),
      () =>
        page.locator(
          'button[type="submit"], input[type="submit"], ' +
          'button:has-text("Login"), button:has-text("Sign in")',
        ),
    ],
  );

  // Wait for login to finish: URL changes away from /login OR a post-login marker appears.
  await Promise.race([
    page.waitForURL((url) => !new RegExp(LOGIN_URL.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'), 'i').test(url.pathname), {{
      timeout: 15000,
    }}),
    page.waitForLoadState('networkidle', {{ timeout: 15000 }}),
  ]).catch(() => {{}});

  await expect(page.locator('input[type="password"]')).toHaveCount(0, {{ timeout: 5000 }});
}}

async function fillFirst(page, label, candidates, value) {{
  for (const fn of candidates) {{
    if (!fn) continue;
    try {{
      const loc = fn().first();
      await loc.waitFor({{ state: 'visible', timeout: 2500 }});
      await loc.fill(value);
      return;
    }} catch {{
      /* try next */
    }}
  }}
  throw new Error(
    `Login helper could not find the ${{label}} field. Set LOGIN_${{label
      .toUpperCase()
      .replace(/[^A-Z]/g, '_')}}_CSS env var with a CSS selector that matches the field.`,
  );
}}

async function clickFirst(page, label, candidates) {{
  for (const fn of candidates) {{
    if (!fn) continue;
    try {{
      const loc = fn().first();
      await loc.waitFor({{ state: 'visible', timeout: 2500 }});
      await loc.click();
      return;
    }} catch {{
      /* try next */
    }}
  }}
  // Final fallback: press Enter on the password field to submit.
  await page.locator('input[type="password"]').first().press('Enter');
  void label;
}}

module.exports = {{ BASE_URL, LOGIN_URL, login }};
"""
    return {"filename": "auth.js", "content": content}


def _spec_file(test_case: TestCase, *, base_url: str | None) -> dict[str, str]:
    title = test_case.scenario or "Scenario"
    filename = f"tests/{_slug(title)}.spec.js"
    needs_login = _needs_login(test_case)
    steps = _strip_leading_login_steps(test_case.steps or []) if needs_login else list(test_case.steps or [])
    has_navigate = any(
        (s.get("action") or "").lower() == "navigate" for s in steps
    )

    lines: list[str] = []
    lines.append("const { test, expect } = require('@playwright/test');")
    lines.append("const { t } = require('../runtime');")
    lines.append("const { BASE_URL, login } = require('../auth');")
    lines.append("")
    lines.append("test.use({ baseURL: BASE_URL });")
    lines.append("")

    if test_case.preconditions:
        lines.append("// Preconditions:")
        for p in test_case.preconditions:
            lines.append(f"//  - {p}")
        lines.append("")

    if needs_login:
        lines.append("test.beforeEach(async ({ page }) => {")
        lines.append("  await login(page);")
        lines.append("});")
        lines.append("")

    lines.append(
        f"test('{_esc_ts(title)}', async ({{ page, request }}) => {{"
    )

    if not has_navigate:
        lines.append("  await t.goto(page, '/');")

    for step in steps:
        lines.append(_step_to_line(step))

    if test_case.expected_results:
        lines.append("")
        lines.append("  // Expected results:")
        for expected in test_case.expected_results:
            lines.append(f"  //  - {_esc_ts(expected)}")

    lines.append("});")
    lines.append("")

    return {"filename": filename, "content": "\n".join(lines)}


def _runtime_file() -> dict[str, str]:
    content = r"""/**
 * runtime.js — generic locator helpers shared by every generated test.
 *
 * Design: tests don't know the target app's DOM, so each helper tries
 * several universal selector strategies in order and returns on the
 * first match. Nothing here is tied to a specific app, domain, or feature.
 *
 * Every failure dumps visible page text to help diagnose mismatches.
 */
const { expect } = require('@playwright/test');

const DEFAULT_TIMEOUT = 15_000;
const STRATEGY_TIMEOUT = 2_000;

// Generic English stopwords used only to pick a "content word" when we fall
// back from a multi-word phrase to a single word. Not tied to any domain.
const STOPWORDS = new Set([
  'the', 'a', 'an', 'of', 'to', 'for', 'in', 'on', 'from', 'by',
  'with', 'at', 'and', 'or', 'is', 'be', 'as',
]);

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Case-insensitive, whitespace-flexible regex for a label.
 * A label of "Two Words" will match "TwoWords", "Two  Words", "TWO WORDS", etc.
 */
function labelPattern(label, exact = false) {
  const words = label.trim().split(/\s+/).filter(Boolean).map(escapeRegex);
  if (words.length === 0) return /.*/;
  const flex = words.join('\\s*');
  return new RegExp(exact ? `^\\s*${flex}\\s*$` : flex, 'i');
}

/**
 * Derive progressively looser variants of a multi-word label using only its
 * word structure — no synonym dictionary, no domain knowledge.
 *
 *   "Alpha Beta Gamma" ->
 *     ["Alpha Beta Gamma", "Alpha Beta", "Beta Gamma", "Alpha", "Beta", "Gamma"]
 */
function labelVariants(label) {
  const raw = label.trim();
  if (!raw) return [raw];
  const out = new Set([raw]);
  const words = raw.split(/\s+/).filter(Boolean);
  for (let i = words.length - 1; i >= 1; i--) {
    out.add(words.slice(0, i).join(' '));
    out.add(words.slice(words.length - i).join(' '));
  }
  for (const w of words) {
    if (!STOPWORDS.has(w.toLowerCase()) && w.length > 1) out.add(w);
  }
  return Array.from(out);
}

/**
 * Region scoping uses ARIA landmark roles and their semantic HTML equivalents
 * plus the two most common class-name conventions (sidebar/navbar, header/topbar,
 * modal/dialog). These are industry-standard UI conventions, not app-specific.
 */
function scope(page, name) {
  if (!name) return page;
  const map = {
    nav: 'nav, aside, [role="navigation"], [class*="sidebar" i], [class*="nav" i]',
    header: 'header, [role="banner"], [class*="header" i], [class*="topbar" i]',
    main: 'main, [role="main"]',
    dialog: '[role="dialog"], [role="alertdialog"], [class*="modal" i], [class*="dialog" i]',
  };
  const selector = map[name] ?? name;
  return page.locator(selector).first();
}

async function tryStrategy(locatorFn, action, timeout) {
  try {
    const loc = locatorFn().first();
    await loc.waitFor({ state: 'visible', timeout });
    await action(loc);
    return true;
  } catch {
    return false;
  }
}

async function dumpVisibleText(page) {
  try {
    const texts = await page.locator('body *:visible').allTextContents();
    const unique = [
      ...new Set(texts.map((t) => t.trim()).filter((t) => t.length > 0 && t.length < 60)),
    ];
    return unique.slice(0, 30).join(' | ');
  } catch {
    return '<could not capture>';
  }
}

/**
 * If a dialog is now open AND contains a button whose accessible name matches
 * any variant of the label just clicked, click it. Pure structural match —
 * no knowledge of specific actions.
 */
/**
 * "any/first/a <something> row|item|record|entry|card" sentinel — when a
 * generated test asks to click an unspecific row of a list page, we click
 * the first real (non-header) row. Lets list-page tests run without
 * hand-coded record IDs.
 */
const FIRST_ROW_SENTINEL = /^\s*(any|first|a|the\s+first|the\s+next)\b.*\b(row|record|entry|item|card|line|listing|result)s?\b\.?\s*$/i;

function isFirstRowSentinel(label) {
  return FIRST_ROW_SENTINEL.test(label || '');
}

function firstRowCandidates(page) {
  return [
    () => page.locator('table tbody tr').first(),
    () => page.locator('[role="rowgroup"] [role="row"]').first(),
    () => page.getByRole('row').nth(1),
    () => page.getByRole('row').first(),
    () => page.locator('tr:not(:has(th))').first(),
    () => page.getByRole('listitem').first(),
    () => page.locator('[role="listitem"], li').first(),
    () => page.locator('[class*="row" i]:not([class*="header" i])').first(),
    () => page.locator('[class*="card" i]').first(),
  ];
}

async function confirmDialogFollowThrough(page, variants) {
  const dialog = page.locator('[role="dialog"], [role="alertdialog"]').first();
  const isOpen = await dialog.isVisible().catch(() => false);
  if (!isOpen) return;

  // Normalize by lowercasing and stripping all whitespace. Handles the
  // universal compact-vs-spaced mismatch ("Logout" <-> "Log Out",
  // "Signup" <-> "Sign Up", "Checkout" <-> "Check Out", etc.) without
  // any word-specific dictionary.
  const norm = (s) => s.toLowerCase().replace(/\s+/g, '');
  const targets = new Set(variants.map(norm).filter((s) => s.length > 0));

  const buttons = dialog.getByRole('button');
  const count = await buttons.count().catch(() => 0);
  for (let i = 0; i < count; i++) {
    const btn = buttons.nth(i);
    const name = (await btn.textContent().catch(() => ''))?.trim() ?? '';
    if (!name) continue;
    if (targets.has(norm(name))) {
      try {
        await btn.click({ force: true });
        await page.waitForLoadState('networkidle', { timeout: 5_000 }).catch(() => {});
        await page.waitForTimeout(200);
        return;
      } catch {
        /* try next */
      }
    }
  }
}

const t = {
  async goto(page, path) {
    await page.goto(path);
    await this.waitSettled(page);
  },

  async waitSettled(page) {
    await page.waitForLoadState('networkidle', { timeout: 8_000 }).catch(() => {});
    await page.waitForTimeout(200);
  },

  async click(page, label, opts = {}) {
    const deadline = Date.now() + (opts.timeout ?? DEFAULT_TIMEOUT);
    const variants = labelVariants(label);
    const scopes = opts.in ? [scope(page, opts.in), page] : [page];

    // "any pending X row" / "first record" / "the next item" — click the first
    // real list row instead of looking for that literal text. Bridges the LLM's
    // "use a structural placeholder" advice (R6) to a real DOM action.
    if (isFirstRowSentinel(label)) {
      for (const fn of firstRowCandidates(page)) {
        if (Date.now() >= deadline) break;
        const ok = await tryStrategy(
          fn,
          async (loc) => {
            await loc.click({ force: true });
          },
          STRATEGY_TIMEOUT,
        );
        if (ok) {
          await this.waitSettled(page);
          return;
        }
      }
    }

    for (const regionScope of scopes) {
      for (const variant of variants) {
        if (Date.now() >= deadline) break;
        const pattern = labelPattern(variant, opts.exact);
        const strategies = [
          () => regionScope.getByRole('link', { name: pattern }),
          () => regionScope.getByRole('button', { name: pattern }),
          () => regionScope.getByRole('menuitem', { name: pattern }),
          () => regionScope.getByRole('tab', { name: pattern }),
          () => regionScope.getByRole('option', { name: pattern }),
          () => regionScope.getByText(pattern).and(page.locator(':visible')),
          () =>
            regionScope
              .getByText(pattern)
              .locator(
                'xpath=ancestor-or-self::*[self::a or self::button or @role="button" or @role="link" or @role="menuitem" or @role="tab" or @role="option" or @tabindex or @onclick][1]',
              ),
          () =>
            regionScope
              .getByText(pattern)
              .locator('xpath=ancestor-or-self::*[self::div or self::span or self::li][1]'),
        ];
        for (const strat of strategies) {
          if (Date.now() >= deadline) break;
          const ok = await tryStrategy(
            strat,
            async (loc) => {
              // Real click event — fires mousedown/mouseup/focus so React and
              // other frameworks that ignore synthetic events still respond.
              // `force: true` skips actionability pauses that break on SPA re-renders.
              await loc.click({ force: true });
            },
            STRATEGY_TIMEOUT,
          );
          if (ok) {
            await this.waitSettled(page);
            // Generic confirmation-dialog follow-through: if the click caused a
            // dialog to open, and that dialog contains a button matching any
            // variant of the label we clicked, click that too. This handles
            // the universal "click X -> modal asks to confirm X -> click X"
            // pattern without knowing anything about the specific action.
            await confirmDialogFollowThrough(page, variants);
            return;
          }
        }
      }
    }

    const visible = await dumpVisibleText(page);
    throw new Error(
      `t.click: could not click "${label}"${opts.in ? ` in ${opts.in}` : ''}.\n` +
        `  Tried variants: ${variants.join(', ')}\n` +
        `  Visible on page: ${visible}`,
    );
  },

  async fill(page, fieldLabel, value) {
    const variants = labelVariants(fieldLabel);
    for (const variant of variants) {
      const pattern = labelPattern(variant);
      const strategies = [
        () => page.getByLabel(pattern),
        () => page.getByPlaceholder(pattern),
        () => page.getByRole('textbox', { name: pattern }),
        () =>
          page.locator(
            `input[name*="${escapeRegex(variant)}" i], input[id*="${escapeRegex(variant)}" i], textarea[name*="${escapeRegex(variant)}" i]`,
          ),
        // Adjacent-label strategy: many SPAs render a <div>/<span> as the
        // visible label and a separate <input> next to it (no <label for=>).
        // Find the first input that follows a visible text node matching
        // the variant — handles the "What?" / "Where?" / "Loss Impact"
        // pattern in custom-styled forms.
        () =>
          page
            .getByText(pattern)
            .and(page.locator(':visible'))
            .locator(
              'xpath=(./following::input | ./following::textarea | ./parent::*//input | ./parent::*//textarea)[1]',
            )
            .first(),
      ];
      for (const strat of strategies) {
        const ok = await tryStrategy(
          strat,
          async (l) => {
            await l.fill(value);
          },
          STRATEGY_TIMEOUT,
        );
        if (ok) return;
      }
    }
    const visible = await dumpVisibleText(page);
    throw new Error(
      `t.fill: could not find field "${fieldLabel}".\n  Visible on page: ${visible}`,
    );
  },

  async select(page, fieldLabel, value) {
    const variants = labelVariants(fieldLabel);
    for (const variant of variants) {
      const pattern = labelPattern(variant);
      const strategies = [
        () => page.getByLabel(pattern),
        () => page.getByRole('combobox', { name: pattern }),
        () =>
          page.locator(
            `select[name*="${escapeRegex(variant)}" i], select[id*="${escapeRegex(variant)}" i]`,
          ),
      ];
      for (const strat of strategies) {
        const ok = await tryStrategy(
          strat,
          async (l) => {
            await l.selectOption(value);
          },
          STRATEGY_TIMEOUT,
        );
        if (ok) return;
      }
    }
    throw new Error(`t.select: could not find select "${fieldLabel}".`);
  },

  /**
   * Tolerant assertion: passes if ANY variant matches a visible heading,
   * visible text, or a URL path slug.
   */
  async see(page, label, timeout = 10_000) {
    const variants = labelVariants(label);
    const slugs = variants
      .map((v) => v.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, ''))
      .filter(Boolean);

    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      const url = page.url().toLowerCase();
      if (slugs.some((s) => url.includes(s))) return;
      for (const variant of variants) {
        const pattern = labelPattern(variant);
        const headingVisible = await page
          .getByRole('heading', { name: pattern })
          .first()
          .isVisible()
          .catch(() => false);
        if (headingVisible) return;
        const textVisible = await page
          .getByText(pattern)
          .and(page.locator(':visible'))
          .first()
          .isVisible()
          .catch(() => false);
        if (textVisible) return;
      }
      await page.waitForTimeout(250);
    }

    const visible = await dumpVisibleText(page);
    throw new Error(
      `t.see: could not see "${label}".\n` +
        `  Tried variants: ${variants.join(', ')}\n` +
        `  Visible on page: ${visible}`,
    );
  },
};

module.exports = { t, expect };
"""
    return {"filename": "runtime.js", "content": content}


def _playwright_config_file() -> dict[str, str]:
    content = """import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
});
"""
    return {"filename": "playwright.config.js", "content": content}


def _package_json_file() -> dict[str, str]:
    content = """{
  "name": "playwright-tests",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "test": "playwright test",
    "report": "playwright show-report"
  },
  "devDependencies": {
    "@playwright/test": "^1.47.0",
    "@types/node": "^20.0.0"
  }
}
"""
    return {"filename": "package.json", "content": content}


def _readme_file(base_url: str | None) -> dict[str, str]:
    base = base_url or "http://your-app.com"
    content = f"""# Playwright Test Project

This is a ready-to-run Playwright test suite. Just extract the ZIP, install
dependencies, set your credentials, and run the tests.

## 1. First-time setup

Open a terminal **inside this folder** (the one that contains `package.json`)
and run:

```bash
npm install
npx playwright install
```

## 2. Set your environment

### macOS / Linux

```bash
export BASE_URL={base}
export APP_USERNAME=your-user
export APP_PASSWORD=your-pass
```

### Windows (PowerShell)

```powershell
$env:BASE_URL="{base}"
$env:APP_USERNAME="your-user"
$env:APP_PASSWORD="your-pass"
```

## 3. Run the tests

```bash
npm test
```

## 4. Open the HTML report

```bash
npm run report
```

## Folder layout

```
.
├── package.json            # dependencies + npm scripts
├── playwright.config.js    # Playwright config (timeouts, reporters, …)
├── auth.js                 # shared login helper used by all tests
├── runtime.js              # shared locator helpers
└── tests/                  # one .spec.js per test scenario
    └── *.spec.js
```

**Do not move files out of this layout** — each test file imports the login
helper from `../auth`.

## If login fails

The login helper auto-detects common field names (email, username,
employee ID). If your app uses something else, override with env vars:

```bash
export LOGIN_URL=/signin                         # default: /login
export LOGIN_USERNAME_CSS='#empIdField'          # CSS selector for username
export LOGIN_PASSWORD_CSS='#passwordField'       # CSS selector for password
export LOGIN_SUBMIT_CSS='button.login-btn'       # CSS selector for submit
```

Find the right selectors by running:

```bash
npx playwright codegen {base}
```

— log in manually, and the Inspector window will print the selectors to use.
"""
    return {"filename": "README.md", "content": content}


# ---------------- public API ----------------

class PlaywrightGenerator:
    def generate(
        self, test_cases: list[TestCase], *, base_url: str | None = None
    ) -> list[dict[str, str]]:
        files: list[dict[str, str]] = [
            _package_json_file(),
            _playwright_config_file(),
            _runtime_file(),
            _auth_helper_file(base_url),
            _readme_file(base_url),
        ]
        files.extend(_spec_file(tc, base_url=base_url) for tc in test_cases)
        return files


_pw_gen: PlaywrightGenerator | None = None


def get_playwright_generator() -> PlaywrightGenerator:
    global _pw_gen
    if _pw_gen is None:
        _pw_gen = PlaywrightGenerator()
    return _pw_gen
