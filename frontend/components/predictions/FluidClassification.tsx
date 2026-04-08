'use client';
import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { downsampleForChart } from '@/lib/utils/downsampling';

const FLUID_COLORS: Record<string, string> = {
  'Background': '#94a3b8',
  'Pay Zone': '#22c55e',
  'Potential Reservoir': '#3b82f6',
};

export function FluidClassification() {
  const { filteredData } = useWellDataStore();
  const { predictions } = usePredictionStore();

  const option = useMemo(() => {
    if (filteredData.length === 0) return {};

    const depths = filteredData.map((d) => d.DEPTH);
    const fluidTrue = filteredData.map((d) => d['FLUID_CLASS'] as string || 'Unknown');
    const fluidPred = predictions.fluid?.predictions || [];

    const fluidCounts = (fluidPred.length > 0 ? fluidPred : fluidTrue).reduce((acc, f) => {
      acc[f] = (acc[f] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const pieData = Object.entries(fluidCounts).map(([name, value]) => ({
      name,
      value,
      itemStyle: { color: FLUID_COLORS[name] || '#888' },
    }));

    return {
      title: {
        text: 'Fluid Classification',
        left: 'center',
        top: 0,
        textStyle: { fontSize: 12 },
      },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: {
        orient: 'vertical',
        right: 10,
        top: 'middle',
        textStyle: { fontSize: 10 },
      },
      series: [
        {
          name: 'Fluid Type',
          type: 'pie',
          radius: ['30%', '60%'],
          center: ['40%', '50%'],
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
  }, [filteredData, predictions.fluid]);

  if (filteredData.length === 0) return null;

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Fluid Distribution</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <ReactECharts
          option={option}
          style={{ height: '250px', width: '100%' }}
          opts={{ renderer: 'canvas' }}
          notMerge={true}
        />
      </CardContent>
    </Card>
  );
}