'use client';
import ReactECharts from 'echarts-for-react';
import { DataQualityReport } from '@/types';

interface MissingValuesChartProps {
  data: DataQualityReport['missing_values'];
  maxItems?: number;
}

export function MissingValuesChart({ data, maxItems = 15 }: MissingValuesChartProps) {
  const chartData = (data || []).slice(0, maxItems);

  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'axis' },
        grid: { left: 150, right: 20, top: 10, bottom: 30 },
        xAxis: {
          type: 'value',
          name: 'Missing %',
          max: 100,
          axisLabel: { fontSize: 10 },
        },
        yAxis: {
          type: 'category',
          data: chartData.map((m) => m.Column?.substring(0, 25) || ''),
          axisLabel: { fontSize: 9 },
        },
        series: [{
          type: 'bar',
          data: chartData.map((m) => ({
            value: m['Missing %'],
            itemStyle: {
              color: m['Missing %'] > 50 ? '#ef4444' :
                     m['Missing %'] > 20 ? '#f97316' : '#eab308',
            },
          })),
          label: { show: true, position: 'right', formatter: '{c}%' },
          barWidth: '60%',
        }],
      }}
      style={{ height: `${Math.max(200, chartData.length * 35)}px`, width: '100%' }}
      opts={{ renderer: 'canvas' }}
    />
  );
}