import { create } from 'zustand';
import { apiClient } from '@/lib/api';

interface Zone {
  depth_start: number;
  depth_end: number;
  zone_type: string;
  confidence: number;
  thickness: number;
  recommendation: string;
  evidence: string[];
}

interface ZoneSummary {
  zones: Zone[];
  summary: {
    total_reservoir_ft: number;
    total_pay_zone_ft: number;
    total_background_ft: number;
  };
}

interface PorosityAnalysis {
  average: number | null;
  quality: string;
  std_deviation?: number;
  distribution: Record<string, number>;
}

interface PressureAnalysis {
  normal: boolean;
  anomaly_detected: boolean;
  overbalance_risk: string;
  average_psi?: number;
  maximum_psi?: number;
  risk_level?: string;
  risk_description?: string;
}

interface Petrophysics {
  porosity_analysis: PorosityAnalysis;
  pressure_analysis: PressureAnalysis;
  reservoir_quality: string;
  hydrocarbon_potential: string;
}

interface DrillingCondition {
  depth: number;
  pore_pressure_psi: number;
  mud_weight_required_ppg: number;
  kick_risk: string;
  recommendation: string;
  warnings: string[];
}

interface Drilling {
  drilling_conditions: DrillingCondition[];
  overall_assessment: string;
  critical_depths: { depth: number; issue: string; pressure: number }[];
  warnings: string[];
}

interface InterpretationState {
  zones: ZoneSummary | null;
  petrophysics: Petrophysics | null;
  drilling: Drilling | null;
  loading: boolean;
  error: string | null;
  
  fetchInterpretation: (data: any[], depthRange?: { min: number; max: number }) => Promise<void>;
  fetchZoneInterpretation: (data: any[], depthRange?: { min: number; max: number }) => Promise<void>;
  fetchPetrophysics: (data: any[], depthRange?: { min: number; max: number }) => Promise<void>;
  fetchDrilling: (data: any[], depthRange?: { min: number; max: number }) => Promise<void>;
  reset: () => void;
}

export const useInterpretationStore = create<InterpretationState>((set) => ({
  zones: null,
  petrophysics: null,
  drilling: null,
  loading: false,
  error: null,

  fetchInterpretation: async (data, depthRange) => {
    set({ loading: true, error: null });
    try {
      console.log('[Interpret] Sending data, keys:', Object.keys(data[0] || {}), 'length:', data.length);
      const response = await apiClient.post('/interpret/all', {
        data,
        depth_range: depthRange,
        include_confidence: true,
      });
      console.log('[Interpret] Response received:', response);
      
      // Handle response - might be wrapped in .data or direct
      const result = response?.data || response;
      console.log('[Interpret] Result:', result);
      
      set({
        zones: result?.zones || null,
        petrophysics: result?.petrophysics || null,
        drilling: result?.drilling || null,
        loading: false,
      });
    } catch (error: any) {
      console.error('[Interpret] Error:', error);
      const errMsg = error.response?.data?.detail || error.message || 'Interpretation failed';
      set({ loading: false, error: errMsg });
    }
  },

  fetchZoneInterpretation: async (data, depthRange) => {
    set({ loading: true, error: null });
    try {
      const response = await apiClient.post('/interpret/zones', {
        data,
        depth_range: depthRange,
        include_confidence: true,
      });
      const result = response?.data || response;
      set({ zones: result, loading: false });
    } catch (error: any) {
      set({ loading: false, error: error.response?.data?.detail || error.message || 'Zone interpretation failed' });
    }
  },

  fetchPetrophysics: async (data, depthRange) => {
    set({ loading: true, error: null });
    try {
      const response = await apiClient.post('/interpret/petrophysics', {
        data,
        depth_range: depthRange,
      });
      const result = response?.data || response;
      set({ petrophysics: result, loading: false });
    } catch (error: any) {
      set({ loading: false, error: error.response?.data?.detail || error.message || 'Petrophysics interpretation failed' });
    }
  },

  fetchDrilling: async (data, depthRange) => {
    set({ loading: true, error: null });
    try {
      const response = await apiClient.post('/interpret/drilling', {
        data,
        depth_range: depthRange,
      });
      const result = response?.data || response;
      set({ drilling: result, loading: false });
    } catch (error: any) {
      set({ loading: false, error: error.response?.data?.detail || error.message || 'Drilling interpretation failed' });
    }
  },

  reset: () => {
    set({ zones: null, petrophysics: null, drilling: null, error: null });
  },
}));