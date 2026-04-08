'use client';
import { useState, useMemo, useCallback } from 'react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { Button } from '@/components/ui/Button';
import { notify } from '@/components/ui/Toast';
import { Columns3, Check, AlertTriangle, ArrowRight } from 'lucide-react';

interface ColumnMapping {
  source: string;
  target: string;
  confidence: number;
}

const TARGET_COLUMNS = [
  { key: 'DEPTH', label: 'Depth', required: true },
  { key: 'Gamma Ray - Corrected gAPI', label: 'Gamma Ray', required: false },
  { key: 'Resistivity Phase - Corrected - 2MHz ohm.m', label: 'Resistivity', required: false },
  { key: 'Bulk Density - Compensated kg/m3', label: 'Bulk Density', required: false },
  { key: 'Neutron Porosity (Sandstone) Euc', label: 'Neutron Porosity', required: false },
  { key: 'ROP for the Bit - Distance Over Time (On Bottom) m/s', label: 'ROP', required: false },
  { key: 'Weight On Bit N', label: 'WOB', required: false },
  { key: 'Surface Torque Average N.m', label: 'Torque', required: false },
  { key: 'Mud Weight In kg/m3', label: 'Mud Weight', required: false },
  { key: 'Corrected Drilling Exponent unitless', label: 'Dxc', required: false },
];

const KEYWORDS: Record<string, string[]> = {
  'DEPTH': ['depth', 'dept', 'md', 'measured depth', 'tvd'],
  'Gamma Ray - Corrected gAPI': ['gamma', 'gr', 'gamma ray', 'sgamma'],
  'Resistivity Phase - Corrected - 2MHz ohm.m': ['resistivity', 'res', 'ild', 'rt', 'rphase'],
  'Bulk Density - Compensated kg/m3': ['density', 'rho', 'rhob', 'bulk density', 'den'],
  'Neutron Porosity (Sandstone) Euc': ['neutron', 'nphi', 'porosity neutron', 'cnl'],
  'ROP for the Bit - Distance Over Time (On Bottom) m/s': ['rop', 'rate of penetration', 'rop_bit'],
  'Weight On Bit N': ['wob', 'weight on bit', 'weightbit'],
  'Surface Torque Average N.m': ['torque', 'surface torque', 'trq'],
  'Mud Weight In kg/m3': ['mud weight', 'mudweight', 'mw', 'mud_wt'],
  'Corrected Drilling Exponent unitless': ['dxc', 'drilling exponent', 'corrected', 'dexponent'],
};

function findBestMatch(sourceCol: string, targetKey: string): number {
  const source = sourceCol.toLowerCase();
  const keywords = KEYWORDS[targetKey] || [];

  for (const kw of keywords) {
    if (source === kw) return 1.0;
    if (source.includes(kw)) return 0.8;
  }
  return 0;
}

function autoMapColumns(sourceColumns: string[]): ColumnMapping[] {
  const mappings: ColumnMapping[] = [];
  const usedTargets = new Set<string>();

  for (const target of TARGET_COLUMNS) {
    let bestMatch = { source: '', confidence: 0 };
    for (const sourceCol of sourceColumns) {
      const score = findBestMatch(sourceCol, target.key);
      if (score > bestMatch.confidence) {
        bestMatch = { source: sourceCol, confidence: score };
      }
    }
    if (bestMatch.confidence > 0.5 && !usedTargets.has(target.key)) {
      mappings.push({ source: bestMatch.source, target: target.key, confidence: bestMatch.confidence });
      usedTargets.add(target.key);
    }
  }
  return mappings;
}

interface ColumnMapperProps {
  sourceColumns: string[];
  onMappingComplete?: (mappings: Record<string, string>) => void;
  className?: string;
}

export function ColumnMapper({ sourceColumns, onMappingComplete, className }: ColumnMapperProps) {
  const [mappings, setMappings] = useState<Record<string, string>>(() => {
    const auto = autoMapColumns(sourceColumns);
    const map: Record<string, string> = {};
    for (const m of auto) {
      map[m.target] = m.source;
    }
    return map;
  });

  const handleMapping = (targetKey: string, sourceCol: string) => {
    setMappings((prev) => {
      const next = { ...prev };
      if (sourceCol === '__none__') {
        delete next[targetKey];
      } else {
        next[targetKey] = sourceCol;
      }
      return next;
    });
  };

  const applyMappings = () => {
    const requiredMapped = TARGET_COLUMNS.filter((t) => t.required).every((t) => mappings[t.key]);
    if (!requiredMapped) {
      notify.error('Please map all required columns (at minimum DEPTH)');
      return;
    }
    notify.success(`Mapped ${Object.keys(mappings).length} columns`);
    onMappingComplete?.(mappings);
  };

  const mappedCount = Object.keys(mappings).length;
  const requiredMapped = TARGET_COLUMNS.filter((t) => t.required).every((t) => mappings[t.key]);

  return (
    <div className={className}>
      <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
        <Columns3 className="h-4 w-4" /> Column Mapping
      </h3>

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {TARGET_COLUMNS.map((target) => {
          const isMapped = !!mappings[target.key];
          return (
            <div key={target.key} className="flex items-center gap-2">
              <div className="w-28 flex-shrink-0">
                <span className={`text-xs ${target.required ? 'font-semibold text-gray-800' : 'text-gray-600'}`}>
                  {target.label}
                  {target.required && <span className="text-red-500 ml-0.5">*</span>}
                </span>
              </div>
              <ArrowRight className="h-3 w-3 text-gray-400 flex-shrink-0" />
              <select
                value={mappings[target.key] || '__none__'}
                onChange={(e) => handleMapping(target.key, e.target.value)}
                className={`flex-1 h-7 px-2 text-xs border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                  isMapped ? 'border-green-300 bg-green-50' : 'border-gray-300'
                }`}
              >
                <option value="__none__">-- not mapped --</option>
                {sourceColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              {isMapped ? (
                <Check className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
              ) : target.required ? (
                <AlertTriangle className="h-3.5 w-3.5 text-amber-500 flex-shrink-0" />
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-gray-500">
          {mappedCount}/{TARGET_COLUMNS.length} mapped
        </span>
        <Button size="sm" onClick={applyMappings} disabled={!requiredMapped}>
          Apply Mapping
        </Button>
      </div>
    </div>
  );
}
