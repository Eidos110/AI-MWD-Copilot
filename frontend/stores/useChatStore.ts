import { create } from 'zustand';
import { ChatMessage } from '@/types';

interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  highlightedDepths: number[];
  addMessage: (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  setTyping: (typing: boolean) => void;
  setHighlightedDepths: (depths: number[]) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isTyping: false,
  highlightedDepths: [],
  addMessage: (msg) =>
    set((s) => ({
      messages: [
        ...s.messages,
        { ...msg, id: Date.now().toString(), timestamp: new Date() },
      ],
    })),
  setTyping: (typing) => set({ isTyping: typing }),
  setHighlightedDepths: (depths) => set({ highlightedDepths: depths }),
  clearMessages: () => set({ messages: [] }),
}));
