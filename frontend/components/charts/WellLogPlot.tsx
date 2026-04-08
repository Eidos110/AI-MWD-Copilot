'use client';
import { useMemo, useRef, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { useUIStore } from '@/stores/useUIStore';
import { FLUID_COLORS } from '@/lib/constants';
import { downsampleForChart } from '@/lib/utils/downsampling';
import { throttleLeading } from '@/lib/utils/debounce';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

export function WellLogPlot() {
  const { filteredData } = useWellDataStore();
  const { predictions } = usePredictionStore();
  const { setCrosshairDepth, showConfidence } = useUIStore();
  const chartRef = useRef<ReactECharts>(null);

  const option = useMemo(() => {
    if (filteredData.length === 0) return {};

    const depths = filteredData.map((d) => d.DEPTH);
    const maxPoints = 3000;

    // Downsample for performance
    const gr = filteredData.map((d) => Number(d['Gamma Ray - Corrected gAPI']) || 0);
    const resistivity = filteredData.map((d) => Number(d['Resistivity Phase - Corrected - 2MHz ohm.m']) || 0);
    const porosity = filteredData.map((d) => Number(d['PHI_COMBINED']) || 0);
    const wob = filteredData.map((d) => Number(d['Weight On Bit N']) || 0);
    const torque = filteredData.map((d) => Number(d['Surface Torque Average N.m']) || 0);
    const pressure = filteredData.map((d) => Number(d['PREDICTED_PORE_PRESSURE_PSI']) || 0);

    const dsGr = downsampleForChart(depths, gr, maxPoints);
    const dsRes = downsampleForChart(depths, resistivity, maxPoints);
    const dsPoro = downsampleForChart(depths, porosity, maxPoints);
    const dsWob = downsampleForChart(depths, wob, maxPoints);
    const dsTorque = downsampleForChart(depths, torque, maxPoints);
    const dsPressure = downsampleForChart(depths, pressure, maxPoints);

    // Fluid classification data
    const fluidData = filteredData.map((d) => {
      const cls = String(d['FLUID_CLASS'] || 'Background');
      const color = FLUID_COLORS[cls] || '#9ca3af';
      return { value: [0, d.DEPTH], itemStyle: { color } };
    }).filter((_, i) => i % Math.max(1, Math.floor(depths.length / maxPoints)) === 0);

    // Prediction data
    const phiPred = predictions.porosity?.predictions || [];
    const ppPred = predictions.pressure?.predictions || [];
    const phiConf = predictions.porosity?.confidence || [];
    const phiLower = predictions.porosity?.intervals?.lower || [];
    const phiUpper = predictions.porosity?.intervals?.upper || [];

    const dsPhiPred = phiPred.length > 0 ? downsampleForChart(depths, phiPred, maxPoints) : { depths: [], values: [] };
    const dsPpPred = ppPred.length > 0 ? downsampleForChart(depths, ppPred, maxPoints) : { depths: [], values: [] };

    // Confidence band data for porosity
    const confBandData = showConfidence && phiLower.length > 0 && phiUpper.length > 0
      ? downsampleForChart(depths, phiLower, maxPoints).depths.map((d, i) => [
          downsampleForChart(depths, phiLower, maxPoints).values[i],
          d,
        ])
      : [];

    const confUpperData = showConfidence && phiUpper.length > 0
      ? downsampleForChart(depths, phiUpper, maxPoints).values
      : [];

    return {
      animation: false,
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: any) => {
          if (!params || params.length === 0) return '';
          const depth = params[0].value[1] || params[0].value;
          setCrosshairDepth(typeof depth === 'number' ? depth : parseFloat(depth));
          let html = `<b>Depth: ${depth}m</b><br/>`;
          params.forEach((p: any) => {
            if (p.seriesName !== 'Confidence Band') {
              html += `${p.marker} ${p.seriesName}: ${typeof p.value === 'number' ? p.value.toFixed(4) : p.value}<br/>`;
            }
          });
          return html;
        },
      },
      grid: [
        { left: 60, right: 60, top: 40, height: '12%' },
        { left: 60, right: 60, top: '18%', height: '12%' },
        { left: 60, right: 60, top: '34%', height: '12%' },
        { left: 60, right: 60, top: '50%', height: '10%' },
        { left: 60, right: 60, top: '64%', height: '12%' },
        { left: 60, right: 60, top: '80%', height: '14%' },
      ],
      xAxis: [
        { type: 'value', name: 'GR (gAPI)', min: 0, max: 150, gridIndex: 0, nameLocation: 'middle', nameGap: 25 },
        { type: 'log', name: 'Res (ohm.m)', gridIndex: 1, nameLocation: 'middle', nameGap: 25 },
        { type: 'value', name: 'Porosity', min: 0, max: 0.4, gridIndex: 2, nameLocation: 'middle', nameGap: 25 },
        { type: 'value', gridIndex: 3, show: false },
        { type: 'value', name: 'Pressure (psi)', gridIndex: 4, nameLocation: 'middle', nameGap: 25 },
        { type: 'value', name: 'WOB/Torque', gridIndex: 5, nameLocation: 'middle', nameGap: 25 },
      ],
      yAxis: [
        { type: 'value', name: 'Depth (m)', inverse: true, gridIndex: 0 },
        { type: 'value', inverse: true, gridIndex: 1 },
        { type: 'value', inverse: true, gridIndex: 2 },
        { type: 'value', inverse: true, gridIndex: 3 },
        { type: 'value', inverse: true, gridIndex: 4 },
        { type: 'value', inverse: true, gridIndex: 5 },
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1, 2, 3, 4, 5], yAxisIndex: [0, 1, 2, 3, 4, 5] },
      ],
      series: [
        // Track 1: Gamma Ray
        {
          name: 'Gamma Ray',
          type: 'line',
          data: dsGr.values.map((v, i) => [v, dsGr.depths[i]]),
          xAxisIndex: 0,
          yAxisIndex: 0,
          lineStyle: { color: '#22c55e', width: 1 },
          showSymbol: false,
        },
        // Track 2: Resistivity
        {
          name: 'Resistivity',
          type: 'line',
          data: dsRes.values.map((v, i) => [Math.max(0.01, v), dsRes.depths[i]]),
          xAxisIndex: 1,
          yAxisIndex: 1,
          lineStyle: { color: '#ef4444', width: 1 },
          showSymbol: false,
        },
        // Track 3: Porosity (true)
        {
          name: 'Porosity (True)',
          type: 'line',
          data: dsPoro.values.map((v, i) => [v, dsPoro.depths[i]]),
          xAxisIndex: 2,
          yAxisIndex: 2,
          lineStyle: { color: '#3b82f6', width: 1 },
          showSymbol: false,
        },
        // Track 3: Porosity (predicted)
        ...(dsPhiPred.values.length > 0 ? [{
          name: 'Porosity (Pred)',
          type: 'line',
          data: dsPhiPred.values.map((v, i) => [v, dsPhiPred.depths[i]]),
          xAxisIndex: 2,
          yAxisIndex: 2,
          lineStyle: { color: '#f97316', width: 1, type: 'dashed' as const },
          showSymbol: false,
        }] : []),
        // Track 4: Fluid Type
        {
          name: 'Fluid Type',
          type: 'scatter',
          data: fluidData,
          xAxisIndex: 3,
          yAxisIndex: 3,
          symbolSize: 8,
          large: true,
        },
        // Track 5: Pore Pressure
        {
          name: 'Pore Pressure',
          type: 'line',
          data: dsPressure.values.map((v, i) => [v, dsPressure.depths[i]]),
          xAxisIndex: 4,
          yAxisIndex: 4,
          lineStyle: { color: '#a855f7', width: 1 },
          showSymbol: false,
        },
        ...(dsPpPred.values.length > 0 ? [{
          name: 'Pressure (Pred)',
          type: 'line',
          data: dsPpPred.values.map((v, i) => [v, dsPpPred.depths[i]]),
          xAxisIndex: 4,
          yAxisIndex: 4,
          lineStyle: { color: '#ec4899', width: 1, type: 'dashed' as const },
          showSymbol: false,
        }] : []),
        // Track 6: WOB
        {
          name: 'WOB',
          type: 'line',
          data: dsWob.values.map((v, i) => [v / 1000, dsWob.depths[i]]),
          xAxisIndex: 5,
          yAxisIndex: 5,
          lineStyle: { color: '#000000', width: 0.8 },
          showSymbol: false,
        },
        // Track 6: Torque
        {
          name: 'Torque',
          type: 'line',
          data: dsTorque.values.map((v, i) => [v / 1000, dsTorque.depths[i]]),
          xAxisIndex: 5,
          yAxisIndex: 5,
          lineStyle: { color: '#6b7280', width: 0.8 },
          showSymbol: false,
        },
      ],
    };
  }, [filteredData, predictions, showConfidence]);

  const throttledSetCrosshair = useMemo(
    () => throttleLeading((depth: number) => setCrosshairDepth(depth), 50),
    [setCrosshairDepth]
  );

  const onEvents = useMemo(() => ({
    mousemove: (params: any) => {
      if (params.value && typeof params.value[1] === 'number') {
        throttledSetCrosshair(params.value[1]);
      }
    },
    datazoom: throttleLeading((params: any) => {
      // Throttle zoom events to prevent excessive re-renders
    }, 100),
  }), [throttledSetCrosshair]);

  if (filteredData.length === 0) {
    return (
      <Card className="h-full flex items-center justify-center">
        <p className="text-gray-500">No data loaded. Upload a file or wait for data.</p>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Well Log Plot</CardTitle>
      </CardHeader>
      <CardContent className="p-0 h-[calc(100%-48px)]">
        <ReactECharts
          ref={chartRef}
          option={option}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'canvas' }}
          onEvents={onEvents}
          notMerge={true}
        />
      </CardContent>
    </Card>
  );
}
