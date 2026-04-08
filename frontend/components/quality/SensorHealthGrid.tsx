'use client';
import { DataQualityReport } from '@/types';
import clsx from 'clsx';

interface SensorHealthGridProps {
  data: DataQualityReport['sensor_health'];
}

export function SensorHealthGrid({ data }: SensorHealthGridProps) {
  const getHealthColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    if (score >= 40) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getHealthLabel = (score: number) => {
    if (score >= 80) return 'Good';
    if (score >= 60) return 'Fair';
    if (score >= 40) return 'Poor';
    return 'Critical';
  };

  return (
    <div className="grid grid-cols-2 gap-2">
      {(data || []).map((sensor, index) => (
        <div
          key={index}
          className="flex items-center gap-3 p-2 rounded-lg border bg-white"
        >
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate" title={sensor.Sensor}>
              {sensor.Sensor}
            </p>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={clsx('h-full transition-all', getHealthColor(sensor['Health Score (0-100)']))}
                  style={{ width: `${sensor['Health Score (0-100)']}%` }}
                />
              </div>
              <span className="text-xs font-medium w-10 text-right">
                {sensor['Health Score (0-100)']}%
              </span>
            </div>
          </div>
          <span
            className={clsx(
              'px-2 py-0.5 rounded text-xs font-medium',
              sensor['Health Score (0-100)'] >= 80 ? 'bg-green-100 text-green-700' :
              sensor['Health Score (0-100)'] >= 60 ? 'bg-yellow-100 text-yellow-700' :
              sensor['Health Score (0-100)'] >= 40 ? 'bg-orange-100 text-orange-700' :
              'bg-red-100 text-red-700'
            )}
          >
            {getHealthLabel(sensor['Health Score (0-100)'])}
          </span>
        </div>
      ))}
    </div>
  );
}