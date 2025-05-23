import { Geist, Geist_Mono } from "next/font/google";
import { Space_Grotesk } from 'next/font/google';

import "./globals.css";

import { AuthProvider } from "@/contexts/AuthContext";
import { GeneralMessageProvider } from "@/contexts/GeneralMessageContext";
import { AIMessageProvider } from "@/contexts/AIMessageContext";

import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({ subsets: ['latin'] });

export const metadata = {
  title: "TaxQueryAI",
  description: "Get expert guidance on property tax in India for the period 2013-2018. Use our chatbot to explore tax rates, payment methods, exemptions, and legal updates from this timeframe. Simplify your tax queries with instant, accurate responses!",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${spaceGrotesk.className} antialiased bg-white text-black box-border scrollbar-thin scrollbar-thumb-zinc-900 scrollbar-track-zinc-700`}
      >
        <ToastContainer
          position="bottom-right"
          hideProgressBar={true}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          pauseOnHover
          autoClose={3000}
          toastClassName='custom-toast'
        />
        <AuthProvider>
          <GeneralMessageProvider>
            <AIMessageProvider>
              {children}
            </AIMessageProvider>
          </GeneralMessageProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
