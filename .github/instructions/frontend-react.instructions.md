---
applyTo: "**/*.js, **/*.ts, **/*.tsx, **/*.jsx"
---

# React Frontend Development Guidelines

<!--
SECTION PURPOSE: Introduce mandatory frontend guidance for React projects.
PROMPTING: Clear headings; concise bullets for scanability.
COMPLIANCE: Treat rules below as defaults unless project overrides exist.
-->

<CRITICAL_REQUIREMENT type="MANDATORY">
- Use TypeScript for all new components and code. Interfaces MUST define component props.
- Use React 18+ with hooks (no class components for new code).
- Enforce accessibility: semantic HTML first; ARIA enhances, not substitutes.
- Write tests for all new components and logic changes (render tests + key interactions).
- Use Tailwind CSS or CSS Modules. No inline styles.
</CRITICAL_REQUIREMENT>

## Tech Stack Recommendations

### Build & Runtime
- **Framework**: React 18+
- **Build Tool**: Vite (recommended for fast development) or Next.js (for SSR/API routes)
- **Package Manager**: npm, yarn, or pnpm
- **Node Version**: 18+ LTS

### State Management
- **Local State**: React hooks (useState, useReducer)
- **Global State**: Zustand (lightweight, recommended) or Redux Toolkit (complex apps)
- **Server State**: TanStack Query (React Query) for API data

### Styling
- **Tailwind CSS** (recommended): utility-first, responsive, dark mode support
- **CSS Modules**: scoped styles, zero-runtime
- **Styled Components**: CSS-in-JS, component-based

### Testing
- **Unit/Component**: Vitest + Testing Library (recommended) or Jest + Testing Library
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

1. **Props**: Define with TypeScript interfaces; document required vs optional. Provide sensible defaults.
2. **Component Structure**:
   - Use functional components with hooks
   - Keep components small and focused (single responsibility)
   - Extract complex logic into custom hooks
   - Use composition over inheritance

3. **State Management**:
   - Use `useState` for local component state
   - Use `useReducer` for complex state logic
   - Use Context for theme, user, authentication (not for frequent updates)
   - Use TanStack Query or Zustand for global state

4. **Side Effects**: Use `useEffect` sparingly; clean up subscriptions and timers

### Component Template

```typescript
import React from 'react';

interface StockChartProps {
  symbol: string;
  period?: 'daily' | 'weekly' | 'monthly';
  onLoadComplete?: () => void;
}

export const StockChart: React.FC<StockChartProps> = ({
  symbol,
  period = 'daily',
  onLoadComplete,
}) => {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    fetchChartData();
  }, [symbol, period]);

  const fetchChartData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`/api/stocks/${symbol}/chart?period=${period}`);
      const result = await response.json();
      setData(result);
      onLoadComplete?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load chart');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="p-4">Loading chart...</div>;
  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="w-full h-96 border rounded-lg">
      {/* Chart rendering logic */}
    </div>
  );
};

export default StockChart;
```

## Hooks Development

Create custom hooks for reusable logic:

```typescript
// hooks/useStockData.ts
import { useQuery } from '@tanstack/react-query';

export const useStockData = (symbol: string) => {
  return useQuery({
    queryKey: ['stock', symbol],
    queryFn: async () => {
      const response = await fetch(`/api/stocks/${symbol}`);
      return response.json();
    },
  });
};

// Usage in component
const { data, isLoading, error } = useStockData('AAPL');
```

## State Management

### For Local Component State
```typescript
const [count, setCount] = useState(0);
const [user, setUser] = useState<User | null>(null);
```

### For Complex State
```typescript
interface State {
  filters: StockFilter[];
  selectedStocks: Stock[];
  sorting: SortOption;
}

const initialState: State = { /* ... */ };

const [state, dispatch] = useReducer(reducer, initialState);

// In reducer:
dispatch({ type: 'ADD_FILTER', payload: newFilter });
```

### For Global State (Zustand)
```typescript
import { create } from 'zustand';

interface TradingStore {
  portfolio: Stock[];
  addStock: (stock: Stock) => void;
  removeStock: (id: string) => void;
}

export const useTradingStore = create<TradingStore>((set) => ({
  portfolio: [],
  addStock: (stock) => set((state) => ({ 
    portfolio: [...state.portfolio, stock] 
  })),
  removeStock: (id) => set((state) => ({
    portfolio: state.portfolio.filter(s => s.id !== id)
  })),
}));
```

## Styling with Tailwind CSS

```typescript
// Always use className, never inline styles
export const Card: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
    <h2 className="text-xl font-bold text-gray-900 mb-4">{title}</h2>
    {children}
  </div>
);

// Responsive design
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
  {/* Content */}
</div>

// Dark mode (if configured)
<div className="bg-white dark:bg-slate-900 text-gray-900 dark:text-white">
  {/* Content */}
</div>
```

## Error Handling & Error Boundaries

```typescript
// Error Boundary
interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-100 border border-red-300 rounded">
          <h2 className="text-red-800">Something went wrong</h2>
          <p className="text-red-600">{this.state.error?.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <StockDashboard />
</ErrorBoundary>
```

## API Integration

Use a centralized API client for all backend communication:

```typescript
// services/api.ts
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const apiClient = {
  stocks: {
    get: (symbol: string) => fetch(`${API_BASE_URL}/api/stocks/${symbol}`).then(r => r.json()),
    list: () => fetch(`${API_BASE_URL}/api/stocks`).then(r => r.json()),
    getChart: (symbol: string, period: string) => 
      fetch(`${API_BASE_URL}/api/stocks/${symbol}/chart?period=${period}`).then(r => r.json()),
  },
  predictions: {
    get: (symbol: string) => fetch(`${API_BASE_URL}/api/predictions/${symbol}`).then(r => r.json()),
  },
};

// Or use Axios for more advanced features:
import axios from 'axios';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error.response?.data?.message || error.message);
    return Promise.reject(error);
  }
);
```

## Testing

1. **Unit/Component Tests (Vitest + Testing Library)**
   - Test rendering and user interactions
   - Mock API calls and external dependencies
   - Use `render()` and query utilities (`getByRole`, `getByLabelText`, etc.)

2. **Example Test**
   ```typescript
   import { render, screen } from '@testing-library/react';
   import userEvent from '@testing-library/user-event';
   import { StockChart } from './StockChart';

   describe('StockChart', () => {
     it('should render chart when data loads', async () => {
       render(<StockChart symbol="AAPL" />);
       expect(screen.getByText(/loading/i)).toBeInTheDocument();
       // Wait for data to load...
       expect(await screen.findByText(/chart/i)).toBeInTheDocument();
     });

     it('should display error message on fetch failure', async () => {
       // Mock API to reject
       render(<StockChart symbol="INVALID" />);
       expect(await screen.findByText(/error/i)).toBeInTheDocument();
     });
   });
   ```

3. **E2E Tests (Playwright)**
   - Test complete user workflows
   - Test across browsers
   - Use page objects pattern

   ```typescript
   import { test, expect } from '@playwright/test';

   test('user can view stock chart', async ({ page }) => {
     await page.goto('http://localhost:5173');
     await page.click('text=AAPL');
     await page.waitForSelector('[data-testid="chart"]');
     const chart = await page.locator('[data-testid="chart"]');
     await expect(chart).toBeVisible();
   });
   ```

4. **Coverage Policy**
   - Follow `.github/copilot-instructions.md#quality-policy`
   - Target ≥ 85% coverage for UI components
   - Ensure 100% coverage for critical paths (authentication, data entry validation)

## Accessibility Standards

- ✅ Use semantic HTML: `<button>`, `<nav>`, `<main>`, `<form>`
- ✅ Include `<label>` for all form inputs
- ✅ Keyboard navigation: all interactive elements accessible via Tab
- ✅ Screen reader support: use ARIA attributes when semantic HTML insufficient
- ✅ Color contrast: minimum 4.5:1 for text
- ✅ Focus indicators: visible focus state on interactive elements

```typescript
// Good: semantic + label
<label htmlFor="stock-input">Stock Symbol:</label>
<input id="stock-input" type="text" placeholder="AAPL" />

// Bad: no label, no semantics
<div onClick={() => {}}>Click me</div>
```

## Performance Optimization

1. **Lazy Loading**: Use `React.lazy()` and `Suspense` for routes and heavy components
   ```typescript
   const StockAnalysis = React.lazy(() => import('./pages/StockAnalysis'));
   ```

2. **Memoization**: Use `React.memo`, `useMemo`, `useCallback`
   ```typescript
   const PriceDisplay = React.memo(({ price }: { price: number }) => {
     return <div>${price.toFixed(2)}</div>;
   });
   ```

3. **Code Splitting**: Bundle heavy libraries separately
4. **Image Optimization**: Use modern formats (WebP), responsive sizes
5. **Minimize Re-renders**: Keep props stable, use proper dependency arrays

## Naming Conventions

- **Components**: PascalCase (e.g., `StockChart`, `PriceDisplay`)
- **Hooks**: camelCase starting with `use` (e.g., `useStockData`, `useTradingStore`)
- **Events**: camelCase with `on` prefix (e.g., `onClick`, `onLoadComplete`)
- **CSS Classes**: kebab-case or use Tailwind utilities
- **Types/Interfaces**: PascalCase (e.g., `StockData`, `ChartProps`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`, `DEFAULT_PERIOD`)

<PROCESS_REQUIREMENTS type="MANDATORY">
- Run `eslint` and `prettier` locally before committing
- Run component tests with `vitest run` or `jest`
- Run E2E tests with `playwright test` before pushing
- Include accessibility checks: labels, keyboard nav, focus order
- Avoid `any` type; if unavoidable, comment with reason and TODO
</PROCESS_REQUIREMENTS>

<!-- © Capgemini 2025 -->
