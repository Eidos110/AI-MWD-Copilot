'use client';
import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { downsampleForChart } from '@/lib/utils/downsampling';

export function PressurePlot() {
  const { filteredData } = useWellDataStore();
  const { predictions } = usePredictionStore();

  const option = useMemo(() => {
    if (filteredData.length === 0) return {};

    const depths = filteredData.map((d) => d.DEPTH);
    const pressure = filteredData.map((d) => Number(d['PREDICTED_PORE_PRESSURE_PSI']) || 0);
    const mudWeight = filteredData.map((d) => {
      const mw = Number(d['Mud Weight In kg/m3']) || 0;
      return mw > 0 ? mw * 0.000145038 * 1000 : 0; // rough Pa to psi conversion
    });

    const ds = downsampleForChart(depths, pressure, 1000);
    const dsMw = downsampleForChart(depths, mudWeight, 1000);
    const ppPred = predictions.pressure?.predictions || [];
    const dsPred = ppPred.length > 0 ? downsampleForChart(depths, ppPred, 1000) : { depths: [], values: [] };

    return {
      title: {
        text: 'Pore Pressure',
        left: 'center',
        top: 0,
        textStyle: { fontSize: 12 },
      },
      tooltip: { trigger: 'axis' },
      grid: { left: 50, right: 20, top: 30, bottom: 30 },
      xAxis: { type: 'value', name: 'PSI', nameLocation: 'middle', nameGap: 25 },
      yAxis: { type: 'value', name: 'Depth (m)', inverse: true },
      series: [
        {
          name: 'PP (True)',
          type: 'line',
          data: ds.values.map((v, i) => [v, ds.depths[i]]),
          lineStyle: { color: '#a855f7', width: 1.5 },
          showSymbol: false,
        },
        ...(dsPred.values.length > 0 ? [{
          name: 'PP (Pred)',
          type: 'line',
          data: dsPred.values.map((v, i) => [v, dsPred.depths[i]]),
          lineStyle: { color: '#ec4899', width: 1, type: 'dashed' as const },
          showSymbol: false,
        }] : []),
      ],
    };
  }, [filteredData, predictions.pressure]);

  if (filteredData.length === 0) return null;

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Pressure Profile</CardTitle>
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
