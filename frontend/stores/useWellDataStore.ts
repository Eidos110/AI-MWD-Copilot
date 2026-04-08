import { create } from 'zustand';
import { WellDataRow, DepthRange } from '@/types';

interface WellDataState {
  rawData: WellDataRow[];
  filteredData: WellDataRow[];
  columns: string[];
  depthRange: DepthRange;
  uploadedFileName: string | null;
  isLoading: boolean;
  error: string | null;
  setData: (data: WellDataRow[], columns: string[]) => void;
  setDepthRange: (range: DepthRange) => void;
  filterByDepth: () => void;
  setUploadedFileName: (name: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useWellDataStore = create<WellDataState>((set, get) => ({
  rawData: [],
  filteredData: [],
  columns: [],
  depthRange: { min: 2000, max: 2500 },
  uploadedFileName: null,
  isLoading: false,
  error: null,
  setData: (data, columns) => {
    set({ rawData: data, columns, error: null });
    get().filterByDepth();
  },
  setDepthRange: (range) => {
    set({ depthRange: range });
    get().filterByDepth();
  },
  filterByDepth: () => {
    const { rawData, depthRange } = get();
    const filtered = rawData.filter(
      (row) => row.DEPTH >= depthRange.min && row.DEPTH <= depthRange.max
    );
    set({ filteredData: filtered });
  },
  setUploadedFileName: (name) => set({ uploadedFileName: name }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}));
