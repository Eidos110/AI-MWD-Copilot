'use client';
import { AlertTriangle, Info, XCircle, CheckCircle } from 'lucide-react';
import clsx from 'clsx';

export type AnomalyType = 'warning' | 'error' | 'info' | 'success';

export interface AnomalyAlertProps {
  type: AnomalyType;
  title: string;
  message: string;
  depths?: number[];
  onDismiss?: () => void;
}

const ICONS = {
  warning: AlertTriangle,
  error: XCircle,
  info: Info,
  success: CheckCircle,
};

const STYLES = {
  warning: {
    container: 'bg-amber-50 border-amber-200 text-amber-800',
    icon: 'text-amber-500',
    title: 'text-amber-900',
  },
  error: {
    container: 'bg-red-50 border-red-200 text-red-800',
    icon: 'text-red-500',
    title: 'text-red-900',
  },
  info: {
    container: 'bg-blue-50 border-blue-200 text-blue-800',
    icon: 'text-blue-500',
    title: 'text-blue-900',
  },
  success: {
    container: 'bg-green-50 border-green-200 text-green-800',
    icon: 'text-green-500',
    title: 'text-green-900',
  },
};

export function AnomalyAlert({ type, title, message, depths, onDismiss }: AnomalyAlertProps) {
  const Icon = ICONS[type];
  const style = STYLES[type];

  return (
    <div className={clsx('border rounded-lg p-3 flex gap-3', style.container)}>
      <Icon className={clsx('h-5 w-5 flex-shrink-0 mt-0.5', style.icon)} />
      <div className="flex-1 min-w-0">
        <h4 className={clsx('font-medium text-sm', style.title)}>{title}</h4>
        <p className="text-xs mt-1">{message}</p>
        {depths && depths.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {depths.slice(0, 3).map((d, i) => (
              <span
                key={i}
                className={clsx('inline-block px-1.5 py-0.5 rounded text-xs', style.container)}
              >
                {d.toFixed(0)}m
              </span>
            ))}
            {depths.length > 3 && (
              <span className="text-xs opacity-75">+{depths.length - 3} more</span>
            )}
          </div>
        )}
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="flex-shrink-0 opacity-50 hover:opacity-100 transition"
        >
          <XCircle className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}