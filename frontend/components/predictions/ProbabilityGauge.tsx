'use client';
import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { useUIStore } from '@/stores/useUIStore';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { FLUID_COLORS } from '@/lib/constants';

export function ProbabilityGauge() {
  const { predictions } = usePredictionStore();
  const { crosshairDepth } = useUIStore();
  const { filteredData } = useWellDataStore();

  const option = useMemo(() => {
    if (!predictions.fluid || predictions.fluid.probabilities.length === 0) {
      return {
        title: { text: 'No fluid data', left: 'center', top: 'center', textStyle: { fontSize: 12, color: '#999' } },
      };
    }

    // Find closest depth index
    let idx = 0;
    if (crosshairDepth !== null && filteredData.length > 0) {
      let minDist = Infinity;
      filteredData.forEach((d, i) => {
        const dist = Math.abs(d.DEPTH - crosshairDepth);
        if (dist < minDist) { minDist = dist; idx = i; }
      });
      idx = Math.min(idx, predictions.fluid.probabilities.length - 1);
    }

    const probs = predictions.fluid.probabilities[idx] || [0.33, 0.33, 0.34];
    const classes = predictions.fluid.classes || ['Background', 'Pay Zone', 'Potential Reservoir'];
    const depth = filteredData[idx]?.DEPTH?.toFixed(1) || 'N/A';

    return {
      title: {
        text: `Fluid @ ${depth}m`,
        left: 'center',
        top: 0,
        textStyle: { fontSize: 12 },
      },
      tooltip: { trigger: 'item', formatter: '{b}: {d}%' },
      series: [
        {
          type: 'pie',
          radius: ['40%', '70%'],
          center: ['50%', '55%'],
          label: { show: true, fontSize: 10, formatter: '{b}\n{d}%' },
          data: classes.map((cls, i) => ({
            name: cls,
            value: (probs[i] * 100).toFixed(1),
            itemStyle: { color: FLUID_COLORS[cls] || '#999' },
          })),
        },
      ],
    };
  }, [predictions.fluid, crosshairDepth, filteredData]);

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Fluid Probability</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <ReactECharts
          option={option}
          style={{ height: '200px', width: '100%' }}
          opts={{ renderer: 'canvas' }}
          notMerge={true}
        />
      </CardContent>
    </Card>
  );
}
