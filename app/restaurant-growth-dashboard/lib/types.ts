export type RestaurantRecord = {
  restaurantid: string;
  restaurantname: string;
  cuisinetype: string;
  segment: string;
  subregion: string;
  selected_cluster?: string | number;
  cluster?: string | number;
  cluster_label_name: string;
  cluster_description: string;
  gpi_score: number;
  gpi_band: string;
  gpi_rank: number;
  strategy_recommendation: string;
  recommendation_reason: string;
  total_revenue: number;
  total_net_profit: number;
  aggregator_dependence: number;
  cost_discipline_score: number;
  expansion_headroom: number;
  revenue_quality_score: number;
  scale_score: number;
  delivery_revenue_mix: number;
  instore_reliance: number;
  ue_share?: number;
  dd_share?: number;
  sd_share?: number;
  instoreshare?: number;
  pca_1?: number;
  pca_2?: number;
  [key: string]: string | number | undefined;
};

export type DashboardSummary = Record<string, string | number | null | undefined>;

export type ClusterSummary = {
  selected_cluster?: string | number;
  cluster?: string | number;
  cluster_label_name: string;
  restaurant_count: number;
  avg_gpi_score: number;
  avg_total_revenue?: number;
  avg_total_net_profit: number;
  cluster_description?: string;
  [key: string]: string | number | undefined;
};

export type DashboardData = {
  restaurants: RestaurantRecord[];
  summary: DashboardSummary;
  clusterSummary: ClusterSummary[];
  topRestaurants: RestaurantRecord[];
};
