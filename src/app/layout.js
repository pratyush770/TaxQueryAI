import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "TaxQueryAI",
  description: "Get expert guidance on property tax in India for the period 2013-2018. Use our chatbot to explore tax rates, payment methods, exemptions, and legal updates from this timeframe. Simplify your tax queries with instant, accurate responses!",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-white text-black box-border scrollbar-thin scrollbar-thumb-zinc-700`}
      >
        {children}
      </body>
    </html>
  );
}
