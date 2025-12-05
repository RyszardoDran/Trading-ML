<!--
SECTION PURPOSE: Frontmatter defines scope (which files are governed by these rules).
PROMPTING: Keep the YAML minimal; AI should respect this glob when proposing edits.
-->
---
applyTo: "**/*.js, **/*.ts, **/*.tsx, **/*.jsx"
---

# React Frontend Development Guidelines

<!--
SECTION PURPOSE: Introduce mandatory frontend guidance.
PROMPTING: Clear headings; concise bullets for scanability.
COMPLIANCE: Treat rules below as defaults unless project overrides exist.
-->

<CRITICAL_REQUIREMENT type="MANDATORY">
- Use TypeScript for all new components and code. Interfaces MUST define component props.
- Use React 18+ with hooks (no class components for new code).
- Enforce accessibility: semantic HTML first; ARIA enhances, not substitutes.
- Write tests for all new components and logic changes (render tests + key interactions).
- Use CSS-in-JS solution (Tailwind CSS recommended) or CSS Modules. No inline styles.
</CRITICAL_REQUIREMENT>

## Tech Stack Recommendations

### Build & Runtime
- **Framework**: React 18+
- **Build Tool**: Vite (recommended for fast development) or Next.js (for SSR/API routes)
- **Package Manager**: npm, yarn, or pnpm
- **Node Version**: 18+ LTS

### State Management
- **Local State**: React hooks (useState, useReducer)
- **Global State**: Zustand (lightweight) or Redux Toolkit (complex apps)
- **Server State**: TanStack Query (React Query) for API data

### Styling
- **Tailwind CSS** (recommended): utility-first, responsive, dark mode support
- **CSS Modules**: scoped styles, zero-runtime
- **Styled Components**: CSS-in-JS, component-based

### Testing
- **Unit/Component**: Vitest + Vitest UI (faster) or Jest + Testing Library
- **E2E**: Playwright (recommended) or Cypress
- **Visual Testing**: Chromatic or Percy (optional)

### Tooling
- **Linting**: ESLint with React plugin
- **Formatting**: Prettier
- **Type Checking**: TypeScript strict mode

## Project Structure

```
frontend/
├── src/
│   ├── components/           # Reusable UI components
│   │   ├── common/          # Button, Input, Modal, etc.
│   │   ├── layout/          # Header, Sidebar, Footer
│   │   └── features/        # Feature-specific components
│   ├── pages/               # Page components (if using file-based routing)
│   ├── hooks/               # Custom React hooks
│   ├── context/             # React Context providers
│   ├── stores/              # Zustand or Redux stores
│   ├── services/            # API clients, utilities
│   ├── types/               # TypeScript type definitions
│   ├── styles/              # Global styles, Tailwind config
│   ├── utils/               # Helper functions
│   ├── App.tsx              # Root component
│   └── main.tsx             # Entry point
├── tests/                   # Test files
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── public/                  # Static assets
├── vite.config.ts           # Vite configuration (if using Vite)
├── tsconfig.json
├── tailwind.config.js       # Tailwind configuration
├── package.json
└── README.md
```

## General Guidelines

1. **Code Structure**: Build small, single-responsibility components; organize by feature.
2. **TypeScript**: Use strict mode; define all types explicitly. Avoid `any`.
3. **Styling**: Use Tailwind utility classes or CSS Modules. No inline styles.
4. **Accessibility**: Use semantic HTML (`<button>`, `<nav>`, `<main>`); add ARIA only when needed.
5. **Performance**: Lazy-load routes and heavy components; memoize when necessary.
6. **Error Handling**: Wrap components with Error Boundaries; handle API errors explicitly.

## Component Development
3. **API Calls**: Use the shared API client; centralize endpoints and schemas; handle errors explicitly.
4. **Error Boundaries**: Add boundaries around risky trees; fail gracefully.

<!--
SECTION PURPOSE: Make testing guidance explicit and link to SSOTs (Tester chat mode and BDD instructions).
PROMPTING: Reference, don't duplicate. Keep actions concrete for frontend.
-->
## Testing

1. **SSOT References**
	- Tester chat mode: `.github/chatmodes/Tester.chatmode.md`
	- BDD tests instructions: `.github/instructions/bdd-tests.instructions.md`

2. **Unit/UI Tests (default stack: Jest + Testing Library unless overridden)**
	- Cover rendering, critical interactions (click, type, submit), and state transitions.
	- Include accessibility assertions (roles/labels/name, focus management, keyboard nav).
	- Assert async states: loading, success, and error paths; handle empty data gracefully.

3. **E2E/UI Flows (optional, if project uses Playwright/Cypress)**
	- Keep scenarios small and stable; tag appropriately (e.g., `@ui`, `@smoke`).
	- Prefer testids sparingly; select by role/name first.

4. **Coverage Policy**
	- Follow central Quality & Coverage Policy in `.github/copilot-instructions.md#quality-policy`.
	- Ensure hot paths and error paths are fully covered (100%).

<!--
SECTION PURPOSE: Keep apps fast and responsive.
PROMPTING: Short, actionable techniques.
-->
## Performance Optimization

1. **Lazy Loading**: Defer large routes and heavy components.
2. **Memoization**: Use React.memo/useMemo/useCallback to avoid unnecessary work.
3. **Code Splitting**: Split at route and major component boundaries.
4. **Minimize Re-renders**: Keep props stable; use selectors and derived memoized data.

<!--
SECTION PURPOSE: Enforce baseline quality gates.
PROMPTING: XML block for machine-checkable rules.
-->

<PROCESS_REQUIREMENTS type="MANDATORY">
- Run lints and tests locally before PR.
- Include accessibility checks (labels, keyboard nav, focus order) in reviews.
- Avoid `any`; if unavoidable, annotate with a TODO and reason.
</PROCESS_REQUIREMENTS>

<!-- © Capgemini 2025 -->
