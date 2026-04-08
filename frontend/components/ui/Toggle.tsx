'use client';
import clsx from 'clsx';

interface ToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  label?: string;
  size?: 'sm' | 'md';
}

export function Toggle({ enabled, onChange, label, size = 'md' }: ToggleProps) {
  return (
    <label className="inline-flex items-center gap-2 cursor-pointer">
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        onClick={() => onChange(!enabled)}
        className={clsx(
          'relative inline-flex shrink-0 cursor-pointer rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
          {
            'bg-primary-500': enabled,
            'bg-gray-300': !enabled,
            'h-5 w-9': size === 'sm',
            'h-6 w-11': size === 'md',
          }
        )}
      >
        <span
          className={clsx(
            'pointer-events-none inline-block transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
            {
              'h-4 w-4': size === 'sm',
              'h-5 w-5': size === 'md',
              'translate-x-4': enabled && size === 'sm',
              'translate-x-5': enabled && size === 'md',
              'translate-x-0.5': !enabled,
            }
          )}
        />
      </button>
      {label && <span className="text-sm text-gray-700">{label}</span>}
    </label>
  );
}
