'use client';
import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { downsampleForChart } from '@/lib/utils/downsampling';

export function PorosityTrack() {
  const { filteredData } = useWellDataStore();
  const { predictions } = usePredictionStore();

  const option = useMemo(() => {
    if (filteredData.length === 0) return {};

    const depths = filteredData.map((d) => d.DEPTH);
    const phiTrue = filteredData.map((d) => Number(d['PHI_COMBINED']) || 0);
    const phiPred = predictions.porosity?.predictions || [];
    const confidence = predictions.porosity?.confidence || [];
    const intervals = predictions.porosity?.intervals;

    const ds = downsampleForChart(depths, phiTrue, 1000);
    const dsPred = phiPred.length > 0 ? downsampleForChart(depths, phiPred, 1000) : { depths: [], values: [] };
    const dsConf = confidence.length > 0 ? downsampleForChart(depths, confidence, 1000) : { depths: [], values: [] };

    const series: any[] = [
      {
        name: 'Porosity (True)',
        type: 'line',
        data: ds.values.map((v, i) => [v, ds.depths[i]]),
        lineStyle: { color: '#3b82f6', width: 1.5 },
        showSymbol: false,
      },
    ];

    if (dsPred.values.length > 0) {
      series.push({
        name: 'Porosity (Pred)',
        type: 'line',
        data: dsPred.values.map((v, i) => [v, dsPred.depths[i]]),
        lineStyle: { color: '#22c55e', width: 1.5, type: 'dashed' as const },
        showSymbol: false,
      });

      if (intervals && intervals.lower.length > 0 && intervals.upper.length > 0) {
        const lower = downsampleForChart(depths, intervals.lower, 500);
        const upper = downsampleForChart(depths, intervals.upper, 500);

        series.push({
          name: 'Confidence Band',
          type: 'line',
          data: lower.values.map((v, i) => [v, lower.depths[i]]),
          lineStyle: { width: 0 },
          areaStyle: { color: 'rgba(34, 197, 94, 0.2)' },
          showSymbol: false,
        });
        series.push({
          name: 'Upper Bound',
          type: 'line',
          data: upper.values.map((v, i) => [v, upper.depths[i]]),
          lineStyle: { width: 0 },
          showSymbol: false,
        });
      }
    }

    return {
      title: {
        text: 'Porosity (PHI)',
        left: 'center',
        top: 0,
        textStyle: { fontSize: 12 },
      },
      tooltip: { trigger: 'axis' },
      grid: { left: 50, right: 20, top: 30, bottom: 30 },
      xAxis: {
        type: 'value',
        name: 'Porosity',
        nameLocation: 'middle',
        nameGap: 25,
        min: 0,
        max: 1,
      },
      yAxis: { type: 'value', name: 'Depth (m)', inverse: true },
      series,
    };
  }, [filteredData, predictions.porosity]);

  if (filteredData.length === 0) return null;

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Porosity Profile</CardTitle>
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
