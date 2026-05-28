import axios from 'axios';
import { WellDataRow, AllPredictions, DataQualityReport, ShapResult, UploadResponse } from '@/types';

const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || '';

export const apiClient = axios.create({
  baseURL: apiBaseUrl ? `${apiBaseUrl}/api/v1` : '/api/v1',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

export async function getSampleData(minDepth?: number, maxDepth?: number) {
  const params: Record<string, number> = {};
  if (minDepth !== undefined) params.min_depth = minDepth;
  if (maxDepth !== undefined) params.max_depth = maxDepth;
  const { data } = await apiClient.get('/data/sample', { params });
  return data as { columns: string[]; rows: number; data: WellDataRow[]; depth_range: { min: number; max: number } };
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post('/data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function predictAll(
  dataRows: WellDataRow[],
  depthRange?: { min: number; max: number },
  includeConfidence = true
): Promise<AllPredictions> {
  const { data } = await apiClient.post('/predict/all', {
    data: dataRows.slice(0, 500),
    depth_range: depthRange,
    include_confidence: includeConfidence,
  });
  return data;
}

export async function getQualityReport(dataRows: WellDataRow[]): Promise<DataQualityReport> {
  const { data } = await apiClient.post('/quality/report', { data: dataRows.slice(0, 500) });
  return data;
}

export async function getShapExplanation(
  model: 'porosity' | 'fluid' | 'pressure',
  dataRows: WellDataRow[],
  topN = 3
): Promise<ShapResult> {
  const { data } = await apiClient.post('/shap/explain', {
    model,
    data: dataRows.slice(0, 100),
    top_n: topN,
    max_samples: 100,
  });
  return data;
}

export async function getInterpretation(
  dataRows: any[],
  depthRange?: { min: number; max: number }
): Promise<any> {
  const { data } = await apiClient.post('/interpret/all', {
    data: dataRows.slice(0, 500),
    depth_range: depthRange,
    include_confidence: true,
  });
  return data;
}
