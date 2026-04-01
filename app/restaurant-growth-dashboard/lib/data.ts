import fs from "node:fs/promises";
import path from "node:path";
import Papa from "papaparse";
import type { ClusterSummary, DashboardData, DashboardSummary, RestaurantRecord } from "./types";

async function resolveDataRoot(): Promise<string> {
  const candidates = [
    path.join(process.cwd(), "data", "processed"),
    path.join(process.cwd(), "..", "..", "data", "processed")
  ];

  for (const candidate of candidates) {
    const testFile = path.join(candidate, "clustered_restaurants.csv");
    try {
      await fs.access(testFile);
      return candidate;
    } catch {
      // try next candidate
    }
  }

  throw new Error(
    "Could not locate data/processed directory. Expected clustered_restaurants.csv near dashboard app or project root."
  );
}

function maybeNumber(value: unknown) {
  if (value === null || value === undefined || value === "") return 0;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

async function parseCsvFile<T>(filePath: string): Promise<T[]> {
  const raw = await fs.readFile(filePath, "utf8");
  const parsed = Papa.parse<T>(raw, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transform: (value) => {
      const trimmed = value?.trim?.() ?? value;
      return String(trimmed);
    }
  });

  return parsed.data.map((row) => {
    const normalized = Object.fromEntries(
      Object.entries(row as Record<string, unknown>).map(([key, value]) => [key, maybeNumber(value)])
    );
    return normalized as T;
  });
}

async function parseJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function normalizeSummaryShape(rawSummary: unknown): DashboardSummary {
  if (!rawSummary || typeof rawSummary !== "object") {
    return {};
  }

  const summary = rawSummary as Record<string, unknown>;
  const cards =
    summary.overall_kpi_summary_cards && typeof summary.overall_kpi_summary_cards === "object"
      ? (summary.overall_kpi_summary_cards as Record<string, unknown>)
      : {};

  return {
    ...summary,
    ...cards
  } as DashboardSummary;
}

function extractSummaryValue(
  summary: DashboardSummary,
  keys: string[],
  fallback: number
) {
  const match = keys.find((key) => summary[key] !== undefined && summary[key] !== null);
  const value = match ? Number(summary[match]) : fallback;
  return Number.isFinite(value) ? value : fallback;
}

export async function getDashboardData(): Promise<DashboardData> {
  const DATA_ROOT = await resolveDataRoot();
  const restaurantsPath = path.join(DATA_ROOT, "clustered_restaurants.csv");
  const summaryPath = path.join(DATA_ROOT, "dashboard_summary.json");
  const topRestaurantsPath = path.join(DATA_ROOT, "top_restaurants.csv");
  const clusterSummaryPath = path.join(DATA_ROOT, "cluster_dashboard_summary.csv");

  const [restaurants, summary, topRestaurants, clusterSummary] = await Promise.all([
    parseCsvFile<RestaurantRecord>(restaurantsPath),
    parseJsonFile<DashboardSummary>(summaryPath, {}),
    parseCsvFile<RestaurantRecord>(topRestaurantsPath).catch(() => []),
    parseCsvFile<ClusterSummary>(clusterSummaryPath).catch(() => [])
  ]);

  const normalizedRestaurants = restaurants.map((restaurant) => ({
    ...restaurant,
    restaurantid: String(restaurant.restaurantid)
  }));

  const normalizedTopRestaurants = topRestaurants.map((restaurant) => ({
    ...restaurant,
    restaurantid: String(restaurant.restaurantid)
  }));

  const normalizedSummary = normalizeSummaryShape(summary);

  const derivedSummary: DashboardSummary = {
    total_restaurants: normalizedRestaurants.length,
    average_gpi:
      extractSummaryValue(normalizedSummary, ["average_gpi", "avg_gpi", "avg_gpi_score"], average(normalizedRestaurants, "gpi_score")),
    high_potential_count:
      extractSummaryValue(
        normalizedSummary,
        ["high_potential_count", "high_potential_restaurants"],
        normalizedRestaurants.filter((r) => r.gpi_band === "High Potential").length
      ),
    scale_aggressively_count:
      extractSummaryValue(
        normalizedSummary,
        ["scale_aggressively_count"],
        normalizedRestaurants.filter((r) => r.strategy_recommendation === "Scale Aggressively").length
      ),
    average_total_revenue:
      extractSummaryValue(normalizedSummary, ["average_total_revenue", "avg_total_revenue"], average(normalizedRestaurants, "total_revenue")),
    average_total_net_profit:
      extractSummaryValue(
        normalizedSummary,
        ["average_total_net_profit", "avg_total_net_profit"],
        average(normalizedRestaurants, "total_net_profit")
      ),
    ...normalizedSummary
  };

  const clusteredSummary = clusterSummary.map((clusterRow) => {
    const clusterName = String(clusterRow.cluster_label_name ?? "");
    const fallbackDescription =
      normalizedRestaurants.find((restaurant) => restaurant.cluster_label_name === clusterName)?.cluster_description ??
      "Strategic archetype group summary.";
    return {
      ...clusterRow,
      cluster:
        clusterRow.cluster ??
        clusterRow.selected_cluster ??
        clusterName,
      cluster_description:
        clusterRow.cluster_description ?? fallbackDescription
    };
  });

  return {
    restaurants: normalizedRestaurants,
    summary: derivedSummary,
    clusterSummary: clusteredSummary,
    topRestaurants: normalizedTopRestaurants.length
      ? normalizedTopRestaurants
      : [...normalizedRestaurants].sort((a, b) => b.gpi_score - a.gpi_score).slice(0, 10)
  };
}

function average<T extends Record<string, unknown>>(rows: T[], key: keyof T) {
  if (!rows.length) return 0;
  const total = rows.reduce((sum, row) => sum + Number(row[key] ?? 0), 0);
  return total / rows.length;
}
