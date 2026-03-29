import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, FileText, Loader2, AlertCircle } from "lucide-react";
import { useAppContext } from "@/context/AppContext";

const AnalysisTab = () => {
  const { setResult, setActiveTab } = useAppContext();
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = async () => {
    setError("");
    if (!text.trim() && !file) {
      setError("Please enter text or upload a file to analyze.");
      return;
    }

    setLoading(true);
    try {
      let res: Response;
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        res = await fetch("http://localhost:5000/predict", { method: "POST", body: formData });
      } else {
        res = await fetch("http://localhost:5000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      }

      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult({
        prediction: data.prediction,
        confidence: data.confidence,
        extractedText: text || undefined,
      });
      setActiveTab("results");
    } catch (err: unknown) {
      setError((err as Error).message || "Failed to connect to the API. Make sure the Flask server is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto max-w-2xl"
      >
        <h2 className="font-heading text-3xl font-bold">Analyze Content</h2>
        <p className="mt-2 text-muted-foreground">Paste text or upload a document to detect AI-generated content.</p>

        <div className="mt-8 space-y-6">
          <div>
            <label className="mb-2 block text-sm font-medium">Paste Text</label>
            <textarea
              rows={8}
              value={text}
              onChange={(e) => { setText(e.target.value); setFile(null); }}
              placeholder="Enter or paste the text you want to analyze..."
              className="w-full resize-none rounded-xl border bg-card p-4 text-sm shadow-card transition-shadow focus:shadow-elevated focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div className="flex items-center gap-3">
            <div className="h-px flex-1 bg-border" />
            <span className="text-xs font-medium text-muted-foreground">OR</span>
            <div className="h-px flex-1 bg-border" />
          </div>

          <div
            onClick={() => fileRef.current?.click()}
            className="flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed bg-card p-8 transition-colors hover:border-primary hover:bg-accent"
          >
            <Upload className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              {file ? (
                <span className="flex items-center gap-2 font-medium text-foreground">
                  <FileText className="h-4 w-4" /> {file.name}
                </span>
              ) : (
                "Click to upload PDF, DOCX, or PPTX"
              )}
            </p>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.docx,.pptx"
              className="hidden"
              onChange={(e) => { setFile(e.target.files?.[0] || null); setText(""); }}
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="gradient-bg flex w-full items-center justify-center gap-2 rounded-xl py-3.5 font-semibold text-primary-foreground shadow-elevated transition-transform hover:scale-[1.02] disabled:opacity-60"
          >
            {loading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" /> Analyzing...
              </>
            ) : (
              "Analyze"
            )}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default AnalysisTab;
