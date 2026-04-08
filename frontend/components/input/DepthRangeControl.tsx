'use client';
import { useState, useEffect } from 'react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { Button } from '@/components/ui/Button';
import { DEPTH_PRESETS } from '@/lib/constants';
import { MapPin } from 'lucide-react';

interface DepthRangeControlProps {
  className?: string;
}

export function DepthRangeControl({ className }: DepthRangeControlProps) {
  const { rawData, depthRange, setDepthRange } = useWellDataStore();

  const dataMin = rawData.length > 0 ? Math.floor(rawData[0].DEPTH) : 0;
  const dataMax = rawData.length > 0 ? Math.ceil(rawData[rawData.length - 1].DEPTH) : 5000;

  const [localMin, setLocalMin] = useState(depthRange.min);
  const [localMax, setLocalMax] = useState(depthRange.max);

  useEffect(() => {
    setLocalMin(depthRange.min);
    setLocalMax(depthRange.max);
  }, [depthRange]);

  const handleMinChange = (value: string) => {
    const num = parseInt(value, 10);
    if (!isNaN(num)) {
      setLocalMin(num);
    }
  };

  const handleMaxChange = (value: string) => {
    const num = parseInt(value, 10);
    if (!isNaN(num)) {
      setLocalMax(num);
    }
  };

  const applyRange = () => {
    const clampedMin = Math.max(dataMin, Math.min(localMin, localMax - 1));
    const clampedMax = Math.min(dataMax, Math.max(localMax, localMin + 1));
    setDepthRange({ min: clampedMin, max: clampedMax });
  };

  const isPresetActive = (preset: { min: number; max: number }) =>
    depthRange.min === preset.min && depthRange.max === preset.max;

  return (
    <div className={className}>
      <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
        <MapPin className="h-4 w-4" /> Depth Range
      </h3>

      {/* Presets */}
      <div className="grid grid-cols-3 gap-1 mb-3">
        {DEPTH_PRESETS.map((p) => (
          <Button
            key={p.label}
            variant={isPresetActive(p) ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setDepthRange({ min: p.min, max: p.max })}
            className="text-xs"
          >
            {p.label}
          </Button>
        ))}
      </div>

      {/* Custom Range Inputs */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500 w-12">Min (m)</label>
          <input
            type="number"
            value={localMin}
            onChange={(e) => handleMinChange(e.target.value)}
            onBlur={applyRange}
            onKeyDown={(e) => e.key === 'Enter' && applyRange()}
            min={dataMin}
            max={dataMax}
            className="flex-1 h-8 px-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500 w-12">Max (m)</label>
          <input
            type="number"
            value={localMax}
            onChange={(e) => handleMaxChange(e.target.value)}
            onBlur={applyRange}
            onKeyDown={(e) => e.key === 'Enter' && applyRange()}
            min={dataMin}
            max={dataMax}
            className="flex-1 h-8 px-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Range Info */}
      {rawData.length > 0 && (
        <p className="text-xs text-gray-400 mt-2">
          Data range: {dataMin}m - {dataMax}m
        </p>
      )}
    </div>
  );
}
