import { AppProvider, useAppContext } from "@/context/AppContext";
import Navbar from "@/components/Navbar";
import HomeTab from "@/components/HomeTab";
import AnalysisTab from "@/components/AnalysisTab";
import ResultsTab from "@/components/ResultsTab";
import ContactTab from "@/components/ContactTab";
import Footer from "@/components/Footer";
import { AnimatePresence, motion } from "framer-motion";

const TabContent = () => {
  const { activeTab } = useAppContext();

  const tabs: Record<string, React.ReactNode> = {
    home: <HomeTab />,
    analysis: <AnalysisTab />,
    results: <ResultsTab />,
    contact: <ContactTab />,
  };

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.2 }}
      >
        {tabs[activeTab]}
      </motion.div>
    </AnimatePresence>
  );
};

const Index = () => (
  <AppProvider>
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="flex-1">
        <TabContent />
      </main>
      <Footer />
    </div>
  </AppProvider>
);

export default Index;
