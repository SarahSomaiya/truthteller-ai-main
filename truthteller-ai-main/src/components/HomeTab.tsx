import { motion } from "framer-motion";
import { FileText, BrainCircuit, BarChart3, Sparkles } from "lucide-react";
import { useAppContext } from "@/context/AppContext";

const features = [
  { icon: FileText, title: "Multi-Format Input", desc: "Upload PDF, DOCX, PPTX or paste text directly for analysis." },
  { icon: BrainCircuit, title: "AI vs Human Detection", desc: "Advanced ML model classifies content origin with high accuracy." },
  { icon: BarChart3, title: "Confidence Score", desc: "Get a clear percentage score for prediction confidence." },
];

const HomeTab = () => {
  const { setActiveTab } = useAppContext();

  return (
    <div className="hero-bg min-h-[calc(100vh-64px)]">
      <div className="container mx-auto px-4 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto max-w-3xl text-center"
        >
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border bg-card px-4 py-1.5 text-sm text-muted-foreground shadow-card">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            Machine Learning Powered
          </div>
          <h1 className="font-heading text-5xl font-bold leading-tight tracking-tight md:text-6xl">
            AI Content{" "}
            <span className="gradient-text">Detection System</span>
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-lg text-muted-foreground">
            Detect whether a given text or document is AI-generated or human-written using advanced machine learning algorithms.
          </p>
          <div className="mt-10 flex items-center justify-center gap-4">
            <button
              onClick={() => setActiveTab("analysis")}
              className="gradient-bg rounded-xl px-8 py-3 font-semibold text-primary-foreground shadow-elevated transition-transform hover:scale-105"
            >
              Start Analysis
            </button>
            <button
              onClick={() => setActiveTab("contact")}
              className="rounded-xl border bg-card px-8 py-3 font-semibold shadow-card transition-colors hover:bg-secondary"
            >
              Learn More
            </button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mx-auto mt-24 grid max-w-4xl gap-6 md:grid-cols-3"
        >
          {features.map((f, i) => (
            <div
              key={i}
              className="rounded-2xl border bg-card p-6 shadow-card transition-shadow hover:shadow-elevated"
            >
              <div className="mb-4 inline-flex rounded-xl bg-accent p-3">
                <f.icon className="h-5 w-5 text-accent-foreground" />
              </div>
              <h3 className="font-heading text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{f.desc}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

export default HomeTab;
