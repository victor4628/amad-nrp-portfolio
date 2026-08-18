import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AMAD-Enhanced Network Risk Parity for S&P 100",
  description: "Dynamic S&P 100 AMAD network risk parity research, diagnostics, and reporting.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
