export interface WellDataRow {
  DEPTH: number;
  [key: string]: number | string;
}

export interface PredictionResult {
  predictions: number[];
  confidence?: number[];
  intervals?: { lower: number[]; upper: number[] };
}

export interface FluidPrediction {
  predictions: string[];
  probabilities: number[][];
  classes: string[];
  confidence?: string[];
}

export interface AllPredictions {
  porosity?: PredictionResult;
  fluid?: FluidPrediction;
  pressure?: PredictionResult;
  report?: Record<string, any>[];
}

export interface TrackConfig {
  id: string;
  name: string;
  type: 'linear' | 'log' | 'fill' | 'scatter';
  columns: string[];
  colors: string[];
  min?: number;
  max?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  highlightedDepths?: number[];
}

export interface DataQualityReport {
  missing_values: { Column: string; 'Missing Count': number; 'Missing %': number }[];
  completeness: { 'Overall Completeness %': number; 'Rows with Complete Data': number; 'Rows with Any Missing': number };
  sensor_health: { Sensor: string; 'Health Score (0-100)': number }[];
  summary: string;
}

export interface ShapResult {
  summary: string;
  top_positive: Record<string, any>[];
  top_negative: Record<string, any>[];
  importance: Record<string, any>[];
}

export interface DepthRange {
  min: number;
  max: number;
}

export interface UploadResponse {
  filename: string;
  columns: string[];
  rows: number;
  preview: WellDataRow[];
  depth_range: DepthRange;
  validation: { valid: boolean; missing_columns: string[]; warnings: string[] };
}
