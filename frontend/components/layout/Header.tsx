'use client';
import { useUIStore } from '@/stores/useUIStore';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { Button } from '@/components/ui/Button';
import { Menu, MessageSquare, Activity, BarChart3, Brain, TrendingUp } from 'lucide-react';

export function Header() {
  const { toggleSidebar, toggleChatSidebar, liveMode, setLiveMode, activeTab, setActiveTab } = useUIStore();
  const { rawData, uploadedFileName } = useWellDataStore();

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between z-10">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={toggleSidebar}>
          <Menu className="h-5 w-5" />
        </Button>
        <div className="flex items-center gap-2">
          <Activity className="h-6 w-6 text-primary-500" />
          <h1 className="text-lg font-bold text-gray-900">AI MWD Copilot</h1>
        </div>
        {uploadedFileName && (
          <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
            {uploadedFileName}
          </span>
        )}
        {rawData.length > 0 && (
          <span className="text-xs text-gray-500">{rawData.length.toLocaleString()} pts</span>
        )}
      </div>

      <nav className="flex items-center gap-1">
        <Button
          variant={activeTab === 'dashboard' ? 'primary' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('dashboard')}
        >
          <BarChart3 className="h-4 w-4 mr-1" />
          Dashboard
        </Button>
        <Button
          variant={activeTab === 'quality' ? 'primary' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('quality')}
        >
          Quality
        </Button>
        <Button
          variant={activeTab === 'shap' ? 'primary' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('shap')}
        >
          <Brain className="h-4 w-4 mr-1" />
          SHAP
        </Button>
        <Button
          variant={activeTab === 'interpret' ? 'primary' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('interpret')}
        >
          <TrendingUp className="h-4 w-4 mr-1" />
          Interpret
        </Button>
      </nav>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Live</span>
          <button
            onClick={() => setLiveMode(!liveMode)}
            className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors ${
              liveMode ? 'bg-green-500' : 'bg-gray-300'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition ${
                liveMode ? 'translate-x-4' : 'translate-x-0.5'
              } translate-y-0.5`}
            />
          </button>
          {liveMode && <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />}
        </div>
        <Button variant="ghost" size="sm" onClick={toggleChatSidebar}>
          <MessageSquare className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
