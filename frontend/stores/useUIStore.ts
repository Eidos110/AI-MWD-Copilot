import { create } from 'zustand';

interface UIState {
  sidebarOpen: boolean;
  chatSidebarOpen: boolean;
  liveMode: boolean;
  crosshairDepth: number | null;
  activeTab: 'dashboard' | 'quality' | 'shap' | 'interpret';
  showConfidence: boolean;
  toggleSidebar: () => void;
  toggleChatSidebar: () => void;
  setLiveMode: (mode: boolean) => void;
  setCrosshairDepth: (depth: number | null) => void;
  setActiveTab: (tab: 'dashboard' | 'quality' | 'shap' | 'interpret') => void;
  setShowConfidence: (show: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  chatSidebarOpen: false,
  liveMode: false,
  crosshairDepth: null,
  activeTab: 'dashboard',
  showConfidence: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  toggleChatSidebar: () => set((s) => ({ chatSidebarOpen: !s.chatSidebarOpen })),
  setLiveMode: (mode) => set({ liveMode: mode }),
  setCrosshairDepth: (depth) => set({ crosshairDepth: depth }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setShowConfidence: (show) => set({ showConfidence: show }),
}));
