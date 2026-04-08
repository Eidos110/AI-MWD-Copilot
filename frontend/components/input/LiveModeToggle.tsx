'use client';
import { Toggle } from '@/components/ui/Toggle';
import { useUIStore } from '@/stores/useUIStore';
import { Radio, FileText } from 'lucide-react';

interface LiveModeToggleProps {
  className?: string;
}

export function LiveModeToggle({ className }: LiveModeToggleProps) {
  const { liveMode, setLiveMode } = useUIStore();

  return (
    <div className={className}>
      <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
        {liveMode ? <Radio className="h-4 w-4 text-green-500" /> : <FileText className="h-4 w-4" />}
        {' '}Analysis Mode
      </h3>

      <div className="space-y-2">
        <Toggle
          enabled={liveMode}
          onChange={setLiveMode}
          label={liveMode ? 'Live Streaming' : 'Post Analysis'}
          size="sm"
        />
        <p className="text-xs text-gray-500">
          {liveMode
            ? 'Receiving real-time data via WebSocket'
            : 'Analyzing uploaded or sample data'}
        </p>
      </div>

      {liveMode && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-green-600">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
          </span>
          Connected
        </div>
      )}
    </div>
  );
}
