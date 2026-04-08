'use client';
import { useRef, useMemo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import clsx from 'clsx';
import { WellDataRow } from '@/types';

interface VirtualTableProps {
  data: WellDataRow[];
  columns?: string[];
  maxHeight?: string;
  rowHeight?: number;
  onRowClick?: (row: WellDataRow, index: number) => void;
  selectedIndex?: number;
  stickyHeader?: boolean;
}

const DEFAULT_COLUMNS = [
  'DEPTH',
  'Gamma Ray - Corrected gAPI',
  'Resistivity Phase - Corrected - 2MHz ohm.m',
  'PHI_COMBINED',
  'FLUID_CLASS',
  'PREDICTED_PORE_PRESSURE_PSI',
];

const COLUMN_LABELS: Record<string, string> = {
  DEPTH: 'Depth (m)',
  'Gamma Ray - Corrected gAPI': 'GR (gAPI)',
  'Resistivity Phase - Corrected - 2MHz ohm.m': 'RT (ohm.m)',
  PHI_COMBINED: 'Porosity',
  FLUID_CLASS: 'Fluid',
  PREDICTED_PORE_PRESSURE_PSI: 'Pressure (psi)',
};

export function VirtualTable({
  data,
  columns = DEFAULT_COLUMNS,
  maxHeight = '400px',
  rowHeight = 36,
  onRowClick,
  selectedIndex,
  stickyHeader = true,
}: VirtualTableProps) {
  const parentRef = useRef<HTMLDivElement>(null);

  const visibleColumns = useMemo(
    () => columns.filter((col) => data.length === 0 || col in data[0]),
    [columns, data]
  );

  const rowVirtualizer = useVirtualizer({
    count: data.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight,
    overscan: 10,
  });

  const formatValue = (value: unknown, column: string): string => {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'number') {
      if (column === 'DEPTH') return value.toFixed(1);
      if (column === 'PHI_COMBINED') return (value * 100).toFixed(2) + '%';
      return value.toFixed(2);
    }
    return String(value);
  };

  const getCellClass = (column: string, value: unknown): string => {
    if (column === 'FLUID_CLASS') {
      switch (value) {
        case 'Pay Zone':
          return 'text-green-600 bg-green-50';
        case 'Potential Reservoir':
          return 'text-blue-600 bg-blue-50';
        default:
          return 'text-gray-600';
      }
    }
    if (column === 'PHI_COMBINED' && typeof value === 'number') {
      if (value > 0.15) return 'text-green-600';
      if (value > 0.08) return 'text-amber-600';
    }
    return '';
  };

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 border rounded">
        No data available
      </div>
    );
  }

  return (
    <div
      ref={parentRef}
      className="border rounded-lg overflow-auto"
      style={{ maxHeight }}
    >
      <table className="w-full text-sm">
        <thead className={clsx('bg-gray-50 border-b', stickyHeader && 'sticky top-0 z-10')}>
          <tr>
            <th className="px-3 py-2 text-left font-medium text-gray-600 w-16">#</th>
            {visibleColumns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap"
              >
                {COLUMN_LABELS[col] || col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colSpan={visibleColumns.length + 1} style={{ height: 0, padding: 0 }}>
              <div
                style={{
                  height: `${rowVirtualizer.getTotalSize()}px`,
                  width: '100%',
                  position: 'relative',
                }}
              >
                {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                  const row = data[virtualRow.index];
                  const isSelected = selectedIndex === virtualRow.index;

                  return (
                    <div
                      key={virtualRow.index}
                      className={clsx(
                        'flex items-center border-b hover:bg-gray-50 cursor-pointer transition-colors',
                        isSelected && 'bg-blue-50'
                      )}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: `${virtualRow.size}px`,
                        transform: `translateY(${virtualRow.start}px)`,
                      }}
                      onClick={() => onRowClick?.(row, virtualRow.index)}
                    >
                      <td className="px-3 py-1 text-gray-400 w-16">
                        {virtualRow.index + 1}
                      </td>
                      {visibleColumns.map((col) => (
                        <td
                          key={col}
                          className={clsx(
                            'px-3 py-1 whitespace-nowrap',
                            getCellClass(col, row[col])
                          )}
                        >
                          {formatValue(row[col], col)}
                        </td>
                      ))}
                    </div>
                  );
                })}
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}