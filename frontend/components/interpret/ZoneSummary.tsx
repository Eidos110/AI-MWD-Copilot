'use client';

import ReactECharts from 'echarts-for-react';
import { useInterpretationStore } from '@/stores/useInterpretationStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

const ZONE_COLORS: Record<string, string> = {
  'Potential Reservoir': '#3b82f6',
  'Pay Zone': '#22c55e',
  'Background': '#94a3b8',
};

export function ZoneSummaryPanel() {
  const { zones, loading, error } = useInterpretationStore();

  if (loading) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Zone Summary</CardTitle>
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
          <CardTitle className="text-sm">Zone Summary</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-red-500">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!zones || !zones.summary) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Zone Summary</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-gray-500">No zone data available</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { summary, zones: zoneList } = zones;

  const pieData = [
    { name: 'Potential Reservoir', value: summary.total_reservoir_ft, itemStyle: { color: ZONE_COLORS['Potential Reservoir'] } },
    { name: 'Pay Zone', value: summary.total_pay_zone_ft, itemStyle: { color: ZONE_COLORS['Pay Zone'] } },
    { name: 'Background', value: summary.total_background_ft, itemStyle: { color: ZONE_COLORS['Background'] } },
  ].filter(d => d.value > 0);

  const option = {
    title: {
      text: 'Zone Distribution',
      left: 'center',
      top: 0,
      textStyle: { fontSize: 12 },
    },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ft ({d}%)' },
    series: [
      {
        name: 'Zone Type',
        type: 'pie',
        radius: ['20%', '50%'],
        center: ['50%', '55%'],
        data: pieData,
        label: { show: false },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
      },
    ],
  };

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Zone Summary</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <ReactECharts
          option={option}
          style={{ height: '200px', width: '100%' }}
          opts={{ renderer: 'canvas' }}
        />
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <div className="font-medium text-blue-600">{summary.total_reservoir_ft} ft</div>
            <div className="text-gray-500">Reservoir</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-green-600">{summary.total_pay_zone_ft} ft</div>
            <div className="text-gray-500">Pay Zone</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-gray-600">{summary.total_background_ft} ft</div>
            <div className="text-gray-500">Background</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ZoneRecommendations() {
  const { zones } = useInterpretationStore();

  if (!zones || !zones.zones || zones.zones.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Zone Recommendations</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {zones.zones.map((zone, idx) => (
            <div
              key={idx}
              className="p-2 rounded border text-xs"
              style={{ borderColor: ZONE_COLORS[zone.zone_type] || '#ccc' }}
            >
              <div className="flex justify-between items-center mb-1">
                <span
                  className="font-medium"
                  style={{ color: ZONE_COLORS[zone.zone_type] || '#000' }}
                >
                  {zone.zone_type}
                </span>
                <span className="text-gray-500">
                  {zone.depth_start.toFixed(0)} - {zone.depth_end.toFixed(0)} ft
                </span>
              </div>
              <div className="text-gray-600">{zone.recommendation}</div>
              {zone.evidence && zone.evidence.length > 0 && (
                <div className="mt-1 text-gray-400 text-xs">
                  Evidence: {zone.evidence.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}