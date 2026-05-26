import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface AnalysisResult {
  prediction: string;
  confidence: number;
  extractedText?: string;
}

interface AppContextType {
  isDark: boolean;
  toggleTheme: () => void;
  result: AnalysisResult | null;
  setResult: (r: AnalysisResult | null) => void;
  activeTab: string;
  setActiveTab: (t: string) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("theme") === "dark";
    }
    return false;
  });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState("home");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
    localStorage.setItem("theme", isDark ? "dark" : "light");
  }, [isDark]);

  const toggleTheme = () => setIsDark((p) => !p);

  return (
    <AppContext.Provider value={{ isDark, toggleTheme, result, setResult, activeTab, setActiveTab }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
};
