import { getDashboardData } from "../lib/data";
import { DashboardShell } from "../components/dashboard-shell";

export default async function DashboardPage() {
  const data = await getDashboardData();
  return <DashboardShell data={data} />;
}
