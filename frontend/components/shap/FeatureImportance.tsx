'use client';
import ReactECharts from 'echarts-for-react';
import { ShapResult } from '@/types';

interface FeatureImportanceProps {
  data: ShapResult['importance'];
  title?: string;
}

export function FeatureImportance({ data, title = 'Feature Importance' }: FeatureImportanceProps) {
  const chartData = (data || []).map((item: any) => ({
    name: item.feature || item.name || '',
    value: Math.abs(item.importance || item.value || 0),
  }));

  chartData.sort((a, b) => b.value - a.value);
  const top10 = chartData.slice(0, 10);

  return (
    <ReactECharts
      option={{
        title: {
          text: title,
          left: 'center',
          textStyle: { fontSize: 12 },
        },
        tooltip: {
          trigger: 'axis',
          formatter: (params: any) => {
            const p = params[0];
            return `${p.name}<br/>Importance: ${p.value.toFixed(4)}`;
          },
        },
        grid: { left: 120, right: 20, top: 35, bottom: 20 },
        xAxis: {
          type: 'value',
          name: 'Importance',
          axisLabel: { fontSize: 10 },
        },
        yAxis: {
          type: 'category',
          data: top10.map((d) => d.name),
          axisLabel: {
            fontSize: 10,
            width: 100,
            overflow: 'truncate',
          },
        },
        series: [{
          type: 'bar',
          data: top10.map((d) => ({
            value: d.value,
            itemStyle: {
              color: '#3b82f6',
              borderRadius: [0, 4, 4, 0],
            },
          })),
          label: { show: true, position: 'right', fontSize: 10 },
          barWidth: '60%',
        }],
      }}
      style={{ height: `${Math.max(200, top10.length * 40)}px`, width: '100%' }}
      opts={{ renderer: 'canvas' }}
    />
  );
}