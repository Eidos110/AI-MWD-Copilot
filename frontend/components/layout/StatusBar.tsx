'use client';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { useUIStore } from '@/stores/useUIStore';
import { usePredictionStore } from '@/stores/usePredictionStore';

export function StatusBar() {
  const { rawData, filteredData, depthRange } = useWellDataStore();
  const { liveMode, crosshairDepth } = useUIStore();
  const { predictions } = usePredictionStore();

  return (
    <footer className="bg-white border-t border-gray-200 px-4 py-1.5 flex items-center justify-between text-xs text-gray-500">
      <div className="flex items-center gap-4">
        <span>Data: {filteredData.length.toLocaleString()} / {rawData.length.toLocaleString()} pts</span>
        <span>Depth: {depthRange.min}m - {depthRange.max}m</span>
        {crosshairDepth !== null && (
          <span className="font-medium text-gray-700">Cursor: {crosshairDepth.toFixed(1)}m</span>
        )}
      </div>
      <div className="flex items-center gap-4">
        {predictions.porosity && <span className="text-green-600">Porosity</span>}
        {predictions.fluid && <span className="text-yellow-600">Fluid</span>}
        {predictions.pressure && <span className="text-purple-600">Pressure</span>}
        {liveMode && (
          <span className="flex items-center gap-1 text-green-600">
            <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
            LIVE
          </span>
        )}
        <span>v3.0.0</span>
      </div>
    </footer>
  );
}
