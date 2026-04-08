'use client';
import { useEffect, useState } from 'react';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { getQualityReport } from '@/lib/api';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import ReactECharts from 'echarts-for-react';

export function QualityDashboard() {
  const { filteredData } = useWellDataStore();
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (filteredData.length === 0) return;
    setLoading(true);
    getQualityReport(filteredData)
      .then(setReport)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [filteredData.length]);

  if (loading) return <LoadingSpinner />;
  if (!report) return <Card className="p-8 text-center text-gray-500">No quality data available</Card>;

  const completeness = report.completeness;
  const healthData = report.sensor_health || [];
  const missingData = (report.missing_values || []).slice(0, 10);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-primary-500">{completeness['Overall Completeness %']}%</p>
            <p className="text-sm text-gray-500">Overall Completeness</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-green-500">{completeness['Rows with Complete Data']}</p>
            <p className="text-sm text-gray-500">Complete Rows</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-red-500">{completeness['Rows with Any Missing']}</p>
            <p className="text-sm text-gray-500">Rows with Missing</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle className="text-sm">Sensor Health</CardTitle></CardHeader>
          <CardContent>
            <ReactECharts
              option={{
                tooltip: { trigger: 'axis' },
                grid: { left: 100, right: 20, top: 10, bottom: 30 },
                xAxis: { type: 'value', max: 100 },
                yAxis: {
                  type: 'category',
                  data: healthData.map((h: any) => h.Sensor),
                  axisLabel: { fontSize: 10 },
                },
                series: [{
                  type: 'bar',
                  data: healthData.map((h: any) => ({
                    value: h['Health Score (0-100)'],
                    itemStyle: {
                      color: h['Health Score (0-100)'] >= 80 ? '#22c55e' :
                             h['Health Score (0-100)'] >= 60 ? '#eab308' : '#ef4444',
                    },
                  })),
                  label: { show: true, position: 'right', formatter: '{c}%' },
                }],
              }}
              style={{ height: `${Math.max(150, healthData.length * 40)}px`, width: '100%' }}
              opts={{ renderer: 'canvas' }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-sm">Missing Values</CardTitle></CardHeader>
          <CardContent>
            <ReactECharts
              option={{
                tooltip: { trigger: 'axis' },
                grid: { left: 150, right: 20, top: 10, bottom: 30 },
                xAxis: { type: 'value', max: 100, name: '%' },
                yAxis: {
                  type: 'category',
                  data: missingData.map((m: any) => m.Column?.substring(0, 25) || ''),
                  axisLabel: { fontSize: 9 },
                },
                series: [{
                  type: 'bar',
                  data: missingData.map((m: any) => m['Missing %']),
                  itemStyle: { color: '#f97316' },
                  label: { show: true, position: 'right', formatter: '{c}%' },
                }],
              }}
              style={{ height: `${Math.max(150, missingData.length * 35)}px`, width: '100%' }}
              opts={{ renderer: 'canvas' }}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle className="text-sm">Quality Summary</CardTitle></CardHeader>
        <CardContent>
          <pre className="text-xs whitespace-pre-wrap text-gray-600">{report.summary}</pre>
        </CardContent>
      </Card>
    </div>
  );
}
