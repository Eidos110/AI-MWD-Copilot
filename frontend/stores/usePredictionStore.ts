import { create } from 'zustand';
import { AllPredictions } from '@/types';

interface PredictionState {
  predictions: AllPredictions;
  isLoading: boolean;
  confidenceThreshold: number;
  setPredictions: (predictions: AllPredictions) => void;
  clearPredictions: () => void;
  setConfidenceThreshold: (threshold: number) => void;
  setLoading: (loading: boolean) => void;
}

export const usePredictionStore = create<PredictionState>((set) => ({
  predictions: {},
  isLoading: false,
  confidenceThreshold: 0.5,
  setPredictions: (predictions) => set({ predictions, isLoading: false }),
  clearPredictions: () => set({ predictions: {} }),
  setConfidenceThreshold: (threshold) => set({ confidenceThreshold: threshold }),
  setLoading: (loading) => set({ isLoading: loading }),
}));
