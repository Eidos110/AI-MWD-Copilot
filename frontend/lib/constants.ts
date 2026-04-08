export const TRACK_CONFIGS = [
  {
    id: 'gr',
    name: 'Gamma Ray',
    column: 'Gamma Ray - Corrected gAPI',
    color: '#22c55e',
    min: 0,
    max: 150,
    unit: 'gAPI',
  },
  {
    id: 'resistivity',
    name: 'Resistivity',
    column: 'Resistivity Phase - Corrected - 2MHz ohm.m',
    color: '#ef4444',
    scale: 'log',
    unit: 'ohm.m',
  },
  {
    id: 'porosity',
    name: 'Porosity',
    column: 'PHI_COMBINED',
    color: '#3b82f6',
    min: 0,
    max: 0.4,
    unit: 'v/v',
  },
  {
    id: 'fluid',
    name: 'Fluid Type',
    column: 'FLUID_CLASS',
    type: 'categorical',
  },
  {
    id: 'pressure',
    name: 'Pore Pressure',
    column: 'PREDICTED_PORE_PRESSURE_PSI',
    color: '#a855f7',
    unit: 'psi',
  },
  {
    id: 'drilling',
    name: 'Drilling Parameters',
    columns: ['Weight On Bit N', 'Surface Torque Average N.m'],
    colors: ['#000000', '#6b7280'],
    unit: 'N / N.m',
  },
];

export const FLUID_COLORS: Record<string, string> = {
  'Background': '#9ca3af',
  'Pay Zone': '#eab308',
  'Potential Reservoir': '#f87171',
};

export const DEPTH_PRESETS = [
  { label: 'Shallow', min: 500, max: 1500 },
  { label: 'Mid', min: 1500, max: 2500 },
  { label: 'Deep', min: 2500, max: 3500 },
];
