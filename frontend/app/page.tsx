'use client';
import { useEffect, useState } from 'react';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { StatusBar } from '@/components/layout/StatusBar';
import { WellLogPlot } from '@/components/charts/WellLogPlot';
import { ChatSidebar } from '@/components/ai-copilot/ChatSidebar';
import { QualityDashboard } from '@/components/quality/QualityDashboard';
import { ShapPanel } from '@/components/shap/ShapPanel';
import { InterpretTab } from '@/components/interpret/InterpretTab';
import { ProbabilityGauge } from '@/components/predictions/ProbabilityGauge';
import { PressurePlot } from '@/components/predictions/PressurePlot';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { useUIStore } from '@/stores/useUIStore';
import { getSampleData, predictAll } from '@/lib/api';

export default function Dashboard() {
  const { rawData, filteredData, isLoading, setData, setLoading, setError, depthRange } = useWellDataStore();
  const { predictions, setPredictions, setLoading: setPredLoading } = usePredictionStore();
  const { activeTab, sidebarOpen, chatSidebarOpen, showConfidence, crosshairDepth } = useUIStore();
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    if (initialized) return;
    setInitialized(true);
    loadDefaultData();
  }, []);

  async function loadDefaultData() {
    setLoading(true);
    try {
      const result = await getSampleData();
      setData(result.data, result.columns);
    } catch (err: any) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }

  async function runPredictions() {
    if (filteredData.length === 0) return;
    setPredLoading(true);
    try {
      const result = await predictAll(filteredData, depthRange, showConfidence);
      setPredictions(result);
    } catch (err) {
      console.error('Prediction failed:', err);
    }
  }

  useEffect(() => {
    if (filteredData.length > 0) {
      runPredictions();
    }
  }, [filteredData.length]);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        {sidebarOpen && <Sidebar />}
        <main className="flex-1 flex flex-col overflow-hidden">
          {isLoading ? (
            <div className="flex-1 flex items-center justify-center">
              <LoadingSpinner size="lg" />
            </div>
          ) : (
            <div className="flex-1 overflow-auto p-4">
              {activeTab === 'dashboard' && (
                <div className="flex gap-4 h-full">
                  <div className="flex-1 min-w-0">
                    <WellLogPlot />
                  </div>
                  <div className="w-72 flex flex-col gap-4 overflow-auto">
                    <ProbabilityGauge />
                    <PressurePlot />
                  </div>
                </div>
              )}
              {activeTab === 'quality' && <QualityDashboard />}
              {activeTab === 'shap' && <ShapPanel />}
              {activeTab === 'interpret' && <InterpretTab />}
            </div>
          )}
          <StatusBar />
        </main>
        {chatSidebarOpen && <ChatSidebar />}
      </div>
    </div>
  );
}
