'use client';
import { useState } from 'react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { getShapExplanation } from '@/lib/api';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ShapResult } from '@/types';
import ReactECharts from 'echarts-for-react';

export function ShapPanel() {
  const { filteredData } = useWellDataStore();
  const [selectedModel, setSelectedModel] = useState<'porosity' | 'fluid' | 'pressure'>('porosity');
  const [result, setResult] = useState<ShapResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleExplain = async () => {
    if (filteredData.length === 0) return;
    setLoading(true);
    try {
      const data = await getShapExplanation(selectedModel, filteredData);
      setResult(data);
    } catch (err) {
      console.error('SHAP failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Model Interpretability (SHAP)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-4">
            {(['porosity', 'fluid', 'pressure'] as const).map((m) => (
              <Button
                key={m}
                variant={selectedModel === m ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => setSelectedModel(m)}
              >
                {m.charAt(0).toUpperCase() + m.slice(1)}
              </Button>
            ))}
            <Button size="sm" onClick={handleExplain} disabled={loading || filteredData.length === 0}>
              {loading ? 'Analyzing...' : 'Explain'}
            </Button>
          </div>

          {loading && <LoadingSpinner />}

          {result && !loading && (
            <div className="space-y-4">
              <div className="text-sm whitespace-pre-wrap">{result.summary}</div>

              {result.importance.length > 0 && (
                <ReactECharts
                  option={{
                    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                    grid: { left: 150, right: 40, top: 10, bottom: 30 },
                    xAxis: { type: 'value', name: 'Mean |SHAP|' },
                    yAxis: {
                      type: 'category',
                      data: result.importance.slice(0, 8).map((i: any) =>
                        (i['Display Name'] || i['Original Name'] || '').substring(0, 25)
                      ),
                      axisLabel: { fontSize: 10 },
                    },
                    series: [{
                      type: 'bar',
                      data: result.importance.slice(0, 8).map((i: any) => ({
                        value: i['Mean |SHAP|'],
                        itemStyle: {
                          color: (i['Direction'] || '').includes('UP') ? '#22c55e' : '#ef4444',
                        },
                      })),
                      label: { show: true, position: 'right', formatter: '{c}' },
                    }],
                  }}
                  style={{ height: '300px', width: '100%' }}
                  opts={{ renderer: 'canvas' }}
                />
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
