'use client';

import ReactECharts from 'echarts-for-react';
import { useInterpretationStore } from '@/stores/useInterpretationStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

const QUALITY_COLORS: Record<string, string> = {
  'Excellent': '#22c55e',
  'Good': '#84cc16',
  'Fair': '#eab308',
  'Poor': '#f97316',
  'Tight': '#ef4444',
};

const RISK_COLORS: Record<string, string> = {
  'HIGH': '#ef4444',
  'ELEVATED': '#f97316',
  'NORMAL': '#22c55e',
  'LOW': '#3b82f6',
};

export function PetrophysicsPanel() {
  const { petrophysics, loading, error } = useInterpretationStore();

  if (loading) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Petrophysical Analysis</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-gray-500">Loading...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Petrophysical Analysis</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-red-500">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!petrophysics) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Petrophysical Analysis</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-gray-500">No data available</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { porosity_analysis, pressure_analysis, reservoir_quality, hydrocarbon_potential } = petrophysics;

  const distData = porosity_analysis?.distribution 
    ? Object.entries(porosity_analysis.distribution).map(([name, value]) => ({
        name,
        value,
        itemStyle: { color: QUALITY_COLORS[name] || '#888' },
      }))
    : [];

  const option = {
    title: {
      text: 'Porosity Quality Distribution',
      left: 'center',
      top: 0,
      textStyle: { fontSize: 10 },
    },
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: ['20%', '50%'],
        center: ['50%', '55%'],
        data: distData,
        label: { show: false },
      },
    ],
  };

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Petrophysical Analysis</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        {porosity_analysis?.average && (
          <div className="mb-2">
            <div className="text-xs text-gray-500">Average Porosity</div>
            <div className="text-lg font-medium">
              {(porosity_analysis.average * 100).toFixed(1)}%
              <span className="text-xs text-gray-500 ml-2">
                ({porosity_analysis.quality})
              </span>
            </div>
          </div>
        )}

        {distData.length > 0 && (
          <ReactECharts
            option={option}
            style={{ height: '150px', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        )}

        <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
          <div className="text-center p-2 rounded bg-gray-50">
            <div className="font-medium">Reservoir Quality</div>
            <div className="text-green-600">{reservoir_quality || 'Unknown'}</div>
          </div>
          <div className="text-center p-2 rounded bg-gray-50">
            <div className="font-medium">HC Potential</div>
            <div
              className={
                hydrocarbon_potential === 'High'
                  ? 'text-green-600'
                  : hydrocarbon_potential === 'Moderate'
                  ? 'text-yellow-600'
                  : 'text-red-600'
              }
            >
              {hydrocarbon_potential || 'Unknown'}
            </div>
          </div>
        </div>

        {pressure_analysis && (
          <div className="mt-2 p-2 rounded border text-xs">
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Pressure Risk:</span>
              <span
                className="font-medium"
                style={{ color: RISK_COLORS[pressure_analysis.risk_level || 'NORMAL'] }}
              >
                {pressure_analysis.risk_level || 'NORMAL'}
              </span>
            </div>
            {pressure_analysis.risk_description && (
              <div className="text-gray-400 text-xs mt-1">
                {pressure_analysis.risk_description}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}