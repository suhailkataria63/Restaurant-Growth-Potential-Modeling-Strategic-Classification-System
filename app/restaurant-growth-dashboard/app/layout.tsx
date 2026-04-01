import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Restaurant Growth Potential Dashboard",
  description: "Executive-grade strategic dashboard for restaurant growth readiness analysis."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
