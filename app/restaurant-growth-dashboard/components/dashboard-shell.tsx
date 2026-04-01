"use client";

import { useMemo, useState, type CSSProperties } from "react";
import {
  BarChart3,
  CircleDollarSign,
  Filter,
  LayoutDashboard,
  Search,
  ShieldAlert,
  Sparkles,
  Target,
  TrendingUp
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { DashboardData, RestaurantRecord } from "../lib/types";
import { formatCurrency, formatNumber, formatPercent } from "../lib/format";

type DashboardShellProps = {
  data: DashboardData;
};

type Filters = {
  search: string;
  subregion: string;
  cuisinetype: string;
  segment: string;
  cluster: string;
  gpiBand: string;
  recommendation: string;
};

type DesignPreset = "navy" | "slate" | "emerald" | "mono";

const DESIGN_PRESETS: Array<{ id: DesignPreset; label: string }> = [
  { id: "navy", label: "Navy Glass" },
  { id: "slate", label: "Slate Light" },
  { id: "emerald", label: "Emerald Ops" },
  { id: "mono", label: "Mono Focus" }
];

const CHART_COLORS: Record<DesignPreset, string[]> = {
  navy: ["#4f46e5", "#14b8a6", "#f59e0b", "#ec4899", "#38bdf8", "#22c55e"],
  slate: ["#2563eb", "#0ea5e9", "#06b6d4", "#14b8a6", "#16a34a", "#f59e0b"],
  emerald: ["#10b981", "#14b8a6", "#22c55e", "#84cc16", "#0ea5e9", "#f59e0b"],
  mono: ["#111827", "#1f2937", "#374151", "#4b5563", "#6b7280", "#9ca3af"]
};

const CHART_UI: Record<DesignPreset, { tick: string; grid: string; legend: string }> = {
  navy: { tick: "#8ea0be", grid: "rgba(255, 255, 255, 0.07)", legend: "#d3def0" },
  slate: { tick: "#334155", grid: "rgba(100, 116, 139, 0.25)", legend: "#0f172a" },
  emerald: { tick: "#a7f3d0", grid: "rgba(94, 234, 212, 0.22)", legend: "#d1fae5" },
  mono: { tick: "#1f2937", grid: "rgba(55, 65, 81, 0.22)", legend: "#111827" }
};

export function DashboardShell({ data }: DashboardShellProps) {
  const [designPreset, setDesignPreset] = useState<DesignPreset>("navy");
  const [filters, setFilters] = useState<Filters>({
    search: "",
    subregion: "All",
    cuisinetype: "All",
    segment: "All",
    cluster: "All",
    gpiBand: "All",
    recommendation: "All"
  });
  const chartColors = CHART_COLORS[designPreset];
  const chartUi = CHART_UI[designPreset];

  const [selectedRestaurantId, setSelectedRestaurantId] = useState<string | null>(
    data.topRestaurants[0]?.restaurantid ?? data.restaurants[0]?.restaurantid ?? null
  );

  const filterOptions = useMemo(
    () => ({
      subregions: getUniqueValues(data.restaurants, "subregion"),
      cuisines: getUniqueValues(data.restaurants, "cuisinetype"),
      segments: getUniqueValues(data.restaurants, "segment"),
      clusters: getUniqueValues(data.restaurants, "cluster_label_name"),
      bands: getUniqueValues(data.restaurants, "gpi_band"),
      recommendations: getUniqueValues(data.restaurants, "strategy_recommendation")
    }),
    [data.restaurants]
  );

  const filteredRestaurants = useMemo(() => {
    return data.restaurants.filter((restaurant) => {
      const search = filters.search.trim().toLowerCase();
      const matchesSearch =
        !search ||
        restaurant.restaurantname.toLowerCase().includes(search) ||
        String(restaurant.restaurantid).toLowerCase().includes(search);

      return (
        matchesSearch &&
        matches(restaurant.subregion, filters.subregion) &&
        matches(restaurant.cuisinetype, filters.cuisinetype) &&
        matches(restaurant.segment, filters.segment) &&
        matches(restaurant.cluster_label_name, filters.cluster) &&
        matches(restaurant.gpi_band, filters.gpiBand) &&
        matches(restaurant.strategy_recommendation, filters.recommendation)
      );
    });
  }, [data.restaurants, filters]);

  const selectedRestaurant =
    filteredRestaurants.find((restaurant) => restaurant.restaurantid === selectedRestaurantId) ??
    data.restaurants.find((restaurant) => restaurant.restaurantid === selectedRestaurantId) ??
    filteredRestaurants[0] ??
    data.restaurants[0];

  const summaryCards = useMemo(() => {
    const avgGpi = average(filteredRestaurants, "gpi_score");
    const avgRevenue = average(filteredRestaurants, "total_revenue");
    const avgProfit = average(filteredRestaurants, "total_net_profit");

    return [
      {
        label: "Restaurants",
        value: formatNumber(filteredRestaurants.length),
        hint: `${formatNumber(Number(data.summary.total_restaurants))} total`,
        icon: LayoutDashboard
      },
      {
        label: "Avg GPI",
        value: formatNumber(avgGpi, 1),
        hint: `${formatNumber(countWhere(filteredRestaurants, (r) => r.gpi_band === "High Potential"))} high potential`,
        icon: TrendingUp
      },
      {
        label: "Scale Candidates",
        value: formatNumber(
          countWhere(filteredRestaurants, (r) => r.strategy_recommendation === "Scale Aggressively")
        ),
        hint: "scale aggressively",
        icon: Sparkles
      },
      {
        label: "Avg Revenue",
        value: formatCurrency(avgRevenue),
        hint: `${formatCurrency(avgProfit)} avg net profit`,
        icon: CircleDollarSign
      }
    ];
  }, [data.summary.total_restaurants, filteredRestaurants]);

  const quickStats = useMemo(
    () => [
      {
        label: "High Potential",
        value: countWhere(filteredRestaurants, (r) => r.gpi_band === "High Potential")
      },
      {
        label: "Aggregator-Heavy",
        value: countWhere(filteredRestaurants, (r) => Number(r.aggregator_dependence) >= 0.75)
      },
      {
        label: "Negative Profit",
        value: countWhere(filteredRestaurants, (r) => Number(r.total_net_profit) <= 0)
      }
    ],
    [filteredRestaurants]
  );

  const clusterDistribution = useMemo(
    () => groupCount(filteredRestaurants, "cluster_label_name"),
    [filteredRestaurants]
  );
  const bandDistribution = useMemo(
    () => groupCount(filteredRestaurants, "gpi_band"),
    [filteredRestaurants]
  );
  const recommendationDistribution = useMemo(
    () => groupCount(filteredRestaurants, "strategy_recommendation"),
    [filteredRestaurants]
  );

  const clusterCards = useMemo(() => {
    const groups = new Map<string, RestaurantRecord[]>();
    filteredRestaurants.forEach((restaurant) => {
      const key = restaurant.cluster_label_name;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)?.push(restaurant);
    });

    return Array.from(groups.entries())
      .map(([clusterName, restaurants]) => ({
        clusterName,
        count: restaurants.length,
        avgGpi: average(restaurants, "gpi_score"),
        avgProfit: average(restaurants, "total_net_profit"),
        recommendation: mode(restaurants, "strategy_recommendation"),
        description: restaurants[0]?.cluster_description ?? "Archetype summary"
      }))
      .sort((a, b) => b.count - a.count);
  }, [filteredRestaurants]);

  const topRanked = useMemo(
    () => [...filteredRestaurants].sort((a, b) => b.gpi_score - a.gpi_score).slice(0, 12),
    [filteredRestaurants]
  );

  const tooltipStyle = TOOLTIP_STYLE[designPreset];

  return (
    <main className={`dashboard-page theme-${designPreset}`}>
      <div className="dashboard-container">
        <header className="card dashboard-hero">
          <div className="hero-left">
            <div className="hero-eyebrow">
              <Target size={14} />
              Restaurant Growth Potential System
            </div>
            <h1>Growth Strategy Dashboard</h1>
            <p>Portfolio health, archetypes, and execution priorities in one view.</p>
          </div>
          <div className="hero-right">
            <div className="theme-switcher">
              <span>Style</span>
              <div className="theme-buttons">
                {DESIGN_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => setDesignPreset(preset.id)}
                    className={designPreset === preset.id ? "active" : ""}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="hero-stats">
              {quickStats.map((item) => (
                <div className="hero-stat" key={item.label}>
                  <div className="hero-stat-label">{item.label}</div>
                  <div className="hero-stat-value">{formatNumber(item.value)}</div>
                </div>
              ))}
            </div>
          </div>
        </header>

        <section className="kpi-grid">
          {summaryCards.map((card) => (
            <article className="card kpi-card" key={card.label}>
              <div className="kpi-head">
                <span>{card.label}</span>
                <card.icon size={18} />
              </div>
              <div className="kpi-value">{card.value}</div>
              <div className="kpi-hint">{card.hint}</div>
            </article>
          ))}
        </section>

        <section className="card filters-card">
          <div className="filters-head">
            <h2>
              <Filter size={16} />
              Filters
            </h2>
            <button
              className="ghost-btn"
              onClick={() =>
                setFilters({
                  search: "",
                  subregion: "All",
                  cuisinetype: "All",
                  segment: "All",
                  cluster: "All",
                  gpiBand: "All",
                  recommendation: "All"
                })
              }
            >
              Reset
            </button>
          </div>

          <div className="search-wrap">
            <Search size={16} />
            <input
              placeholder="Search restaurant or ID"
              value={filters.search}
              onChange={(event) => setFilters((prev) => ({ ...prev, search: event.target.value }))}
            />
          </div>

          <div className="filter-grid">
            <FilterSelect
              label="Subregion"
              value={filters.subregion}
              options={filterOptions.subregions}
              onChange={(value) => setFilters((prev) => ({ ...prev, subregion: value }))}
            />
            <FilterSelect
              label="Cuisine"
              value={filters.cuisinetype}
              options={filterOptions.cuisines}
              onChange={(value) => setFilters((prev) => ({ ...prev, cuisinetype: value }))}
            />
            <FilterSelect
              label="Segment"
              value={filters.segment}
              options={filterOptions.segments}
              onChange={(value) => setFilters((prev) => ({ ...prev, segment: value }))}
            />
            <FilterSelect
              label="Archetype"
              value={filters.cluster}
              options={filterOptions.clusters}
              onChange={(value) => setFilters((prev) => ({ ...prev, cluster: value }))}
            />
            <FilterSelect
              label="GPI Band"
              value={filters.gpiBand}
              options={filterOptions.bands}
              onChange={(value) => setFilters((prev) => ({ ...prev, gpiBand: value }))}
            />
            <FilterSelect
              label="Recommendation"
              value={filters.recommendation}
              options={filterOptions.recommendations}
              onChange={(value) => setFilters((prev) => ({ ...prev, recommendation: value }))}
            />
          </div>
        </section>

        <section className="chart-grid chart-grid-3">
          <ChartCard title="Archetype Mix" subtitle="Cluster distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={clusterDistribution}>
                <CartesianGrid stroke={chartUi.grid} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: chartUi.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: chartUi.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" radius={[10, 10, 0, 0]}>
                  {clusterDistribution.map((item, idx) => (
                    <Cell key={item.name} fill={chartColors[idx % chartColors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="GPI Bands" subtitle="Readiness spread">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bandDistribution}>
                <CartesianGrid stroke={chartUi.grid} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: chartUi.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: chartUi.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" radius={[10, 10, 0, 0]}>
                  {bandDistribution.map((item, idx) => (
                    <Cell key={item.name} fill={chartColors[(idx + 2) % chartColors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Action Mix" subtitle="Recommendation distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={recommendationDistribution}>
                <CartesianGrid stroke={chartUi.grid} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: chartUi.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: chartUi.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" radius={[10, 10, 0, 0]}>
                  {recommendationDistribution.map((item, idx) => (
                    <Cell key={item.name} fill={chartColors[(idx + 1) % chartColors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </section>

        <section className="chart-grid chart-grid-2">
          <ChartCard title="Revenue vs Net Profit" subtitle="Cluster-colored portfolio spread">
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart>
                <CartesianGrid stroke={chartUi.grid} />
                <XAxis
                  dataKey="total_revenue"
                  type="number"
                  tick={{ fill: chartUi.tick, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  name="Revenue"
                />
                <YAxis
                  dataKey="total_net_profit"
                  type="number"
                  tick={{ fill: chartUi.tick, fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  name="Net Profit"
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) => formatCurrency(Number(value ?? 0))}
                  labelFormatter={() => ""}
                />
                <Legend wrapperStyle={{ color: chartUi.legend }} />
                {Array.from(new Set(filteredRestaurants.map((item) => item.cluster_label_name))).map((cluster, idx) => (
                  <Scatter
                    key={cluster}
                    name={cluster}
                    data={filteredRestaurants.filter((item) => item.cluster_label_name === cluster)}
                    fill={chartColors[idx % chartColors.length]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="card detail-card">
            <div className="detail-head">
              <ShieldAlert size={16} />
              <h3>Restaurant Detail</h3>
            </div>

            {selectedRestaurant ? (
              <>
                <div className="detail-main">
                  <div>
                    <div className="detail-id">{selectedRestaurant.restaurantid}</div>
                    <h4>{selectedRestaurant.restaurantname}</h4>
                    <p>
                      {selectedRestaurant.cuisinetype} · {selectedRestaurant.segment} · {selectedRestaurant.subregion}
                    </p>
                  </div>
                  <div className="detail-gpi">GPI {formatNumber(selectedRestaurant.gpi_score, 1)}</div>
                </div>

                <div className="detail-tags">
                  <Tag>{selectedRestaurant.gpi_band}</Tag>
                  <Tag>{selectedRestaurant.cluster_label_name}</Tag>
                  <Tag>{selectedRestaurant.strategy_recommendation}</Tag>
                </div>

                <p className="detail-text">{selectedRestaurant.recommendation_reason}</p>

                <div className="mini-grid">
                  <MiniStat label="Revenue" value={formatCurrency(selectedRestaurant.total_revenue)} />
                  <MiniStat label="Net Profit" value={formatCurrency(selectedRestaurant.total_net_profit)} />
                  <MiniStat
                    label="Aggregator"
                    value={formatPercent(Number(selectedRestaurant.aggregator_dependence) * 100, 1)}
                  />
                  <MiniStat
                    label="Cost Discipline"
                    value={formatPercent(Number(selectedRestaurant.cost_discipline_score) * 100, 1)}
                  />
                </div>
              </>
            ) : (
              <div className="detail-empty">No restaurant selected.</div>
            )}
          </div>
        </section>

        <section className="table-grid">
          <div className="card table-card">
            <div className="table-head">
              <h3>
                <BarChart3 size={16} />
                Top GPI Restaurants
              </h3>
              <span>{formatNumber(topRanked.length)} shown</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Restaurant</th>
                    <th>Archetype</th>
                    <th>GPI</th>
                    <th>Action</th>
                    <th>Revenue</th>
                  </tr>
                </thead>
                <tbody>
                  {topRanked.map((restaurant) => (
                    <tr
                      key={restaurant.restaurantid}
                      onClick={() => setSelectedRestaurantId(restaurant.restaurantid)}
                      className={selectedRestaurant?.restaurantid === restaurant.restaurantid ? "active" : ""}
                    >
                      <td>
                        <div className="td-main">{restaurant.restaurantname}</div>
                        <div className="td-sub">
                          {restaurant.cuisinetype} · {restaurant.subregion}
                        </div>
                      </td>
                      <td>{restaurant.cluster_label_name}</td>
                      <td>{formatNumber(restaurant.gpi_score, 1)}</td>
                      <td>{restaurant.strategy_recommendation}</td>
                      <td>{formatCurrency(restaurant.total_revenue)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="cluster-stack">
            {clusterCards.map((cluster) => (
              <article className="card cluster-card" key={cluster.clusterName}>
                <div className="cluster-title">
                  <h4>{cluster.clusterName}</h4>
                  <span>{formatNumber(cluster.count)}</span>
                </div>
                <p>{cluster.description}</p>
                <div className="cluster-metrics">
                  <div>
                    <span>Avg GPI</span>
                    <strong>{formatNumber(cluster.avgGpi, 1)}</strong>
                  </div>
                  <div>
                    <span>Avg Profit</span>
                    <strong>{formatCurrency(cluster.avgProfit)}</strong>
                  </div>
                  <div>
                    <span>Dominant Action</span>
                    <strong>{cluster.recommendation}</strong>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

function ChartCard({
  title,
  subtitle,
  children
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <article className="card chart-card">
      <div className="chart-head">
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
      {children}
    </article>
  );
}

function FilterSelect({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="filter-item">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="All">All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="mini-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return <span className="tag">{children}</span>;
}

function getUniqueValues(restaurants: RestaurantRecord[], field: keyof RestaurantRecord) {
  return Array.from(new Set(restaurants.map((restaurant) => String(restaurant[field] ?? "")))).filter(Boolean).sort();
}

function matches(value: string, filterValue: string) {
  return filterValue === "All" || value === filterValue;
}

function average(restaurants: RestaurantRecord[], field: keyof RestaurantRecord) {
  if (!restaurants.length) return 0;
  return restaurants.reduce((sum, restaurant) => sum + Number(restaurant[field] ?? 0), 0) / restaurants.length;
}

function countWhere(restaurants: RestaurantRecord[], predicate: (restaurant: RestaurantRecord) => boolean) {
  return restaurants.filter(predicate).length;
}

function mode(restaurants: RestaurantRecord[], field: keyof RestaurantRecord) {
  const counts = restaurants.reduce<Record<string, number>>((acc, row) => {
    const key = String(row[field] ?? "N/A");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const [winner] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] ?? ["N/A", 0];
  return winner;
}

function groupCount(restaurants: RestaurantRecord[], field: keyof RestaurantRecord) {
  const counts = restaurants.reduce<Record<string, number>>((accumulator, restaurant) => {
    const key = String(restaurant[field] ?? "Unknown");
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, {});

  return Object.entries(counts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
}

const TOOLTIP_STYLE: Record<DesignPreset, CSSProperties> = {
  navy: {
    backgroundColor: "rgba(11, 20, 36, 0.96)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: 12,
    color: "#e4ecff"
  },
  slate: {
    backgroundColor: "#ffffff",
    border: "1px solid #cbd5e1",
    borderRadius: 12,
    color: "#0f172a"
  },
  emerald: {
    backgroundColor: "rgba(4, 26, 24, 0.96)",
    border: "1px solid rgba(16, 185, 129, 0.42)",
    borderRadius: 12,
    color: "#d1fae5"
  },
  mono: {
    backgroundColor: "#ffffff",
    border: "1px solid #d1d5db",
    borderRadius: 12,
    color: "#111827"
  }
};
