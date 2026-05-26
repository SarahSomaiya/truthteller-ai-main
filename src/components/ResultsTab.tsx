import { motion } from "framer-motion";
import { Bot, User, AlertTriangle } from "lucide-react";
import { useAppContext } from "@/context/AppContext";

const ResultsTab = () => {
  const { result, setActiveTab } = useAppContext();

  if (!result) {
    return (
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mx-auto flex max-w-md flex-col items-center text-center"
        >
          <div className="rounded-2xl border bg-card p-10 shadow-card">
            <AlertTriangle className="mx-auto mb-4 h-10 w-10 text-muted-foreground" />
            <h3 className="font-heading text-xl font-semibold">No Results Yet</h3>
            <p className="mt-2 text-sm text-muted-foreground">Run an analysis first to see results here.</p>
            <button
              onClick={() => setActiveTab("analysis")}
              className="gradient-bg mt-6 rounded-xl px-6 py-2.5 text-sm font-semibold text-primary-foreground"
            >
              Go to Analysis
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  const isAI = result.prediction.toLowerCase() === "ai";
  const pct = Math.round(result.confidence * 100);

  return (
    <div className="container mx-auto px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto max-w-2xl space-y-8"
      >
        <h2 className="font-heading text-3xl font-bold">Analysis Results</h2>

        <div className="rounded-2xl border bg-card p-8 shadow-elevated">
          <div className="flex items-center gap-4">
            <div className={`rounded-xl p-3 ${isAI ? "bg-destructive/10" : "bg-success/10"}`}>
              {isAI ? (
                <Bot className="h-8 w-8 text-destructive" />
              ) : (
                <User className="h-8 w-8 text-success" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Prediction</p>
              <p className={`font-heading text-2xl font-bold ${isAI ? "text-destructive" : "text-success"}`}>
                {isAI ? "AI-Generated" : "Human-Written"}
              </p>
            </div>
          </div>

          <div className="mt-8">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">Confidence</span>
              <span className="font-heading text-xl font-bold">{pct}%</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-secondary">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                className={`h-full rounded-full ${isAI ? "bg-destructive" : "bg-success"}`}
              />
            </div>
          </div>
        </div>

        {result.extractedText && (
          <div className="rounded-2xl border bg-card p-6 shadow-card">
            <h3 className="mb-3 font-heading text-lg font-semibold">Analyzed Text Preview</h3>
            <p className="max-h-40 overflow-auto text-sm text-muted-foreground leading-relaxed">
              {result.extractedText.slice(0, 1000)}
              {result.extractedText.length > 1000 && "..."}
            </p>
          </div>
        )}

        <button
          onClick={() => setActiveTab("analysis")}
          className="rounded-xl border bg-card px-6 py-2.5 text-sm font-semibold shadow-card transition-colors hover:bg-secondary"
        >
          ← Analyze Another
        </button>
      </motion.div>
    </div>
  );
};

export default ResultsTab;
