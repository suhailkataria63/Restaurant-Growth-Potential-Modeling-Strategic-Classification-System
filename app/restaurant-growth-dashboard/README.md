# Restaurant Growth Potential Dashboard

An executive-grade Next.js dashboard for the **Restaurant Growth Potential Modeling & Strategic Classification System**.

## Stack
- Next.js App Router
- React + TypeScript
- Tailwind CSS
- Recharts
- Papa Parse

## Expected data files
Place your prepared files inside:

```bash
data/processed/
  clustered_restaurants.csv
  dashboard_summary.json
  top_restaurants.csv
  cluster_dashboard_summary.csv
```

The dashboard expects `clustered_restaurants.csv` to include:
- cluster interpretation fields (`cluster_label_name`, `cluster_description`)
- GPI fields (`gpi_score`, `gpi_band`, `gpi_rank`)
- recommendation fields (`strategy_recommendation`, `recommendation_reason`)
- KPI fields (`total_revenue`, `total_net_profit`, `cost_discipline_score`, `aggregator_dependence`, etc.)

Note: channel-related values in this project are stored as ratios (0-1), and are rendered as percentages in the UI.

## Start
```bash
npm install
npm run dev
```

From this repository, run:

```bash
cd app/restaurant-growth-dashboard
npm install
npm run dev
```

The loader auto-detects `data/processed` from either:
- this dashboard folder (`./data/processed`)
- the project root (`../../data/processed`)

## Dashboard modules included
- Executive KPI cards
- Cluster overview
- GPI distribution and rankings
- Strategy recommendation analysis
- Search + filter explorer
- Restaurant detail panel
- Multi-restaurant comparison radar

## Architecture notes
- `app/page.tsx` is a server component that loads local processed files.
- `components/dashboard-shell.tsx` is the interactive client dashboard.
- `lib/data.ts` centralizes CSV/JSON loading and summary derivation.

## Recommended next upgrades
- Add URL-synced filters
- Split sections into route groups: `/restaurants`, `/clusters`, `/compare`
- Add virtualized tables for larger portfolios
- Add mini sparklines or historical trend series if time-series data becomes available
