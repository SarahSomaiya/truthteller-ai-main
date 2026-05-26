import { useState } from "react";
import { motion } from "framer-motion";
import { Send, CheckCircle } from "lucide-react";

const ContactTab = () => {
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    setTimeout(() => setSubmitted(false), 3000);
  };

  return (
    <div className="container mx-auto px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto max-w-lg"
      >
        <h2 className="font-heading text-3xl font-bold">Contact Us</h2>
        <p className="mt-2 text-muted-foreground">Have questions? We'd love to hear from you.</p>

        <form onSubmit={handleSubmit} className="mt-8 space-y-5">
          <div>
            <label className="mb-1.5 block text-sm font-medium">Name</label>
            <input
              required
              type="text"
              placeholder="Your name"
              className="w-full rounded-xl border bg-card px-4 py-3 text-sm shadow-card transition-shadow focus:shadow-elevated focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium">Email</label>
            <input
              required
              type="email"
              placeholder="you@example.com"
              className="w-full rounded-xl border bg-card px-4 py-3 text-sm shadow-card transition-shadow focus:shadow-elevated focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium">Message</label>
            <textarea
              required
              rows={5}
              placeholder="Your message..."
              className="w-full resize-none rounded-xl border bg-card p-4 text-sm shadow-card transition-shadow focus:shadow-elevated focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <button
            type="submit"
            className="gradient-bg flex w-full items-center justify-center gap-2 rounded-xl py-3.5 font-semibold text-primary-foreground shadow-elevated transition-transform hover:scale-[1.02]"
          >
            {submitted ? (
              <>
                <CheckCircle className="h-5 w-5" /> Sent!
              </>
            ) : (
              <>
                <Send className="h-4 w-4" /> Send Message
              </>
            )}
          </button>
        </form>
      </motion.div>
    </div>
  );
};

export default ContactTab;
