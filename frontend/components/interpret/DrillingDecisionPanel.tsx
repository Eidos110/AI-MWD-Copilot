'use client';

import { useInterpretationStore } from '@/stores/useInterpretationStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

const RISK_COLORS: Record<string, string> = {
  'HIGH': '#ef4444',
  'ELEVATED': '#f97316',
  'NORMAL': '#22c55e',
  'LOW': '#3b82f6',
};

export function DrillingDecisionPanel() {
  const { drilling, loading, error } = useInterpretationStore();

  if (loading) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Drilling Decision Support</CardTitle>
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
          <CardTitle className="text-sm">Drilling Decision Support</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-red-500">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!drilling) {
    return (
      <Card>
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Drilling Decision Support</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <div className="flex items-center justify-center h-32">
            <span className="text-sm text-gray-500">No data available</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { overall_assessment, warnings, critical_depths = [] } = drilling;

  const hasWarnings = warnings && warnings.length > 0;
  const isAbnormal = overall_assessment?.includes('Abnormal') || overall_assessment?.includes('Monitoring');

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Drilling Decision Support</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <div
          className={`p-2 rounded text-sm text-center mb-2 ${
            isAbnormal
              ? 'bg-red-50 text-red-700 border border-red-200'
              : 'bg-green-50 text-green-700 border border-green-200'
          }`}
        >
          {overall_assessment || 'Normal drilling conditions'}
        </div>

        {critical_depths && critical_depths.length > 0 && (
          <div className="mb-2">
            <div className="text-xs font-medium text-gray-500 mb-1">Critical Depths</div>
            {critical_depths.map((depth, idx) => (
              <div
                key={idx}
                className="p-1 text-xs border border-red-200 rounded bg-red-50"
              >
                <span className="font-medium">{depth.depth} ft</span>
                <span className="text-gray-500 ml-1">- {depth.pressure?.toFixed(0)} psi</span>
              </div>
            ))}
          </div>
        )}

        {hasWarnings && (
          <div className="text-xs">
            <div className="text-xs font-medium text-gray-500 mb-1">Warnings</div>
            <ul className="space-y-1">
              {warnings.slice(0, 5).map((warning, idx) => (
                <li key={idx} className="text-orange-600 text-xs">
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function CriticalDepthsTable() {
  const { drilling } = useInterpretationStore();

  if (!drilling || !drilling.critical_depths || drilling.critical_depths.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Critical Depth Warnings</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1">Depth (ft)</th>
                <th className="text-left py-1">Issue</th>
                <th className="text-right py-1">Pressure (psi)</th>
              </tr>
            </thead>
            <tbody>
              {drilling.critical_depths.map((depth, idx) => (
                <tr key={idx} className="border-b">
                  <td className="py-1">{depth.depth}</td>
                  <td className="py-1">
                    <span
                      className="px-1 py-0.5 rounded text-xs font-medium"
                      style={{
                        backgroundColor:
                          depth.issue === 'HIGH_PRESSURE'
                            ? '#fee2e2'
                            : '#fef3c7',
                        color:
                          depth.issue === 'HIGH_PRESSURE'
                            ? '#dc2626'
                            : '#d97706',
                      }}
                    >
                      {depth.issue === 'HIGH_PRESSURE' ? 'High Pressure' : depth.issue}
                    </span>
                  </td>
                  <td className="py-1 text-right">{depth.pressure?.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}