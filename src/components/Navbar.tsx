import { useAppContext } from "@/context/AppContext";
import { Sun, Moon, BrainCircuit } from "lucide-react";
import { motion } from "framer-motion";

const tabs = [
  { id: "home", label: "Home" },
  { id: "analysis", label: "Analysis" },
  { id: "results", label: "Results" },
  { id: "contact", label: "Contact Us" },
];

const Navbar = () => {
  const { isDark, toggleTheme, activeTab, setActiveTab } = useAppContext();

  return (
    <nav className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur-lg">
      <div className="container mx-auto flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="gradient-bg rounded-lg p-1.5">
            <BrainCircuit className="h-5 w-5 text-primary-foreground" />
          </div>
          <span className="font-heading text-lg font-bold">AI DETECT</span>
        </div>

        <div className="flex items-center gap-1 rounded-xl bg-secondary p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="relative rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              {activeTab === tab.id && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 rounded-lg bg-card shadow-card"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
              )}
              <span className={`relative z-10 ${activeTab === tab.id ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                {tab.label}
              </span>
            </button>
          ))}
        </div>

        <button
          onClick={toggleTheme}
          className="rounded-xl border bg-card p-2.5 transition-colors hover:bg-secondary"
          aria-label="Toggle theme"
        >
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
