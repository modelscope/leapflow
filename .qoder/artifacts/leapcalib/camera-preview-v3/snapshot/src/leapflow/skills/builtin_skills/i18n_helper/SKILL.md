---
name: i18n_helper
description: "Internationalization workflow: string extraction, translation, locale management, and validation"
version: 1.0.0
metadata:
  leapflow:
    category: "productivity"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "productivity"
    tags: ["i18n", "internationalization", "localization", "translation", "l10n", "locale"]
    requires_tools: ["file_read", "file_write", "shell_run"]
platforms: []
triggers:
  - "internationalization"
  - "i18n"
  - "translate"
  - "localization"
  - "国际化"
  - "翻译"
  - "add language support"
  - "extract strings"
---

# i18n Helper

## Purpose

Manage the full internationalization lifecycle: extract user-facing strings from
source code, organize them into locale files, produce translations, and validate
completeness across all supported languages.  This skill treats i18n as a
structured engineering process — not an afterthought bolted onto finished code.

## Guiding Principles

1. **Extract, never hard-code** — Every user-visible string passes through the
   i18n system.  Literal strings in templates, error messages, or UI code are
   defects.
2. **Key naming is API design** — Translation keys are the contract between code
   and translators.  Use semantic, hierarchical keys (`auth.login.button_label`)
   not positional or arbitrary ones (`str_042`).
3. **Context for translators** — A key alone is not enough.  Provide descriptions,
   character limits, and screenshots where the string appears.
4. **Pluralization is not optional** — Different languages have different plural
   rules (1 form for Chinese, 2 for English, 6 for Arabic).  Use ICU
   MessageFormat or the framework's plural system.
5. **Validate continuously** — Missing keys, unused keys, and format-string
   mismatches must be caught in CI, not in production.

## Workflow

### Phase 1 — Audit the Current State

1. Identify the **i18n framework** in use:
   - JavaScript/TypeScript: `i18next`, `react-intl`, `vue-i18n`, `next-intl`.
   - Python: `gettext`, `babel`, `django.utils.translation`.
   - Mobile: `NSLocalizedString` (iOS), `strings.xml` (Android).
   - None: the project has no i18n yet — skip to Phase 2b.
2. Locate **locale files**: `locales/`, `src/i18n/`, `messages/`, `*.po`,
   `*.json`, `*.yaml`, `*.xliff`, `*.arb`.
3. Inventory **supported languages** and their completeness:
   ```
   en: 342 keys (base)
   zh-CN: 338 keys (4 missing)
   ja: 312 keys (30 missing)
   fr: 280 keys (62 missing)
   ```
4. Scan source code for **hard-coded strings** that should be extracted:
   - String literals in JSX/TSX, template files, error messages.
   - User-facing text in CLI output, log messages shown to users, email templates.

### Phase 2a — String Extraction (Existing i18n Setup)

Extract new or changed strings:

1. Run the framework's extraction tool:
   - `i18next-parser`: scans source for `t('key')` calls.
   - `babel extract`: generates `.pot` files from Python source.
   - `formatjs extract`: extracts from `intl.formatMessage()` calls.
2. Compare extracted keys against current base locale file.
3. For each **new key**:
   - Verify the key name follows conventions.
   - Add a description/comment for translator context.
   - Set a default value in the base language.
4. For each **removed key** (no longer in source):
   - Mark as deprecated (do not delete immediately — other branches may
     reference it).
   - Remove in a separate cleanup pass after merge.

### Phase 2b — Bootstrap i18n (No Existing Setup)

When the project has no i18n framework:

1. Choose a framework appropriate to the tech stack.
2. Set up the base configuration (locale detection, fallback chain,
   default namespace).
3. Create the directory structure:
   ```
   src/i18n/
     config.ts       # framework initialization
     locales/
       en/
         common.json  # shared keys
         auth.json    # feature-specific keys
       zh-CN/
         common.json
         auth.json
   ```
4. Replace hard-coded strings in source code with i18n function calls,
   file by file.  Prioritize:
   - UI text visible to end users.
   - Error messages and validation feedback.
   - Email and notification templates.
   - CLI output (if user-facing).

### Phase 3 — Translation

For each target language:

1. Identify **missing keys** by diffing against the base locale.
2. Generate translations following these rules:
   - Preserve **placeholders** exactly (`{name}`, `{{count}}`, `%s`).
   - Respect **plural forms** for the target language.
   - Maintain **HTML tags** and **Markdown** formatting if present.
   - Keep translations **contextually appropriate** — do not translate
     technical terms, brand names, or code identifiers.
   - Respect **character limits** if specified (e.g., button labels).
3. For languages you cannot translate confidently, produce a draft and
   flag it for human review:
   ```json
   {
     "auth.mfa.prompt": {
       "value": "请输入验证码",
       "_review": true,
       "_note": "Machine-translated; needs native review"
     }
   }
   ```

### Phase 4 — Validation

Run comprehensive checks:

1. **Completeness**: every key in the base locale exists in all target locales.
2. **Placeholder integrity**: `{name}` in the base string must appear in every
   translation — missing or extra placeholders cause runtime errors.
3. **Format string safety**: `%d`, `%s` counts must match between base and
   translation.
4. **No untranslated base-language text**: detect translations that are identical
   to the English base (may be untranslated).
5. **ICU syntax validity**: if using MessageFormat, parse each translation
   string for syntax errors.
6. **Unused keys**: keys in locale files that no source code references
   (wasted translator effort and bundle size).
7. **Key sorting**: ensure locale files are sorted alphabetically for
   clean diffs.

Produce a validation report:
```
i18n Validation Report:
  Base language: en (342 keys)
  Languages: zh-CN, ja, fr

  zh-CN: 4 missing keys, 0 placeholder mismatches
    Missing: auth.mfa.backup_codes_title, settings.theme.auto_label, ...

  ja: 30 missing keys, 2 placeholder mismatches
    Mismatch: errors.rate_limit — base has {seconds}, ja translation missing

  fr: 62 missing keys, 0 placeholder mismatches

  Unused keys across all locales: 3
    legacy.old_feature_banner, onboarding.v1_welcome, ...
```

### Phase 5 — Write Back

1. Write updated locale files with `file_write`.
2. Maintain consistent formatting:
   - JSON: 2-space indent, sorted keys, trailing newline.
   - YAML: 2-space indent, no document markers for single docs.
   - PO/POT: standard gettext format.
3. If the project uses a key namespace pattern, place new keys in the
   correct namespace file.

## Error Handling

| Situation | Action |
|---|---|
| Unknown i18n framework | Inspect imports and config files; if unidentifiable, ask the user. |
| Locale files use inconsistent formats (mix of JSON and YAML) | Standardize to the format used by the majority; migrate outliers. |
| Right-to-left language requested (Arabic, Hebrew) | Generate translations; flag that RTL CSS/layout support must be verified separately. |
| String contains complex ICU syntax (select, plural, nested) | Generate carefully; validate with a MessageFormat parser before writing. |

## Limitations

- Translation quality depends on language pair complexity.  East Asian,
  Arabic, and other structurally distant languages should be reviewed by
  native speakers.
- This skill does not modify UI layout for text expansion (German text is
  ~30% longer than English) — that is a design/CSS concern.
- Runtime locale detection and switching logic is framework-specific;
  this skill sets up the data layer, not the runtime behavior.
