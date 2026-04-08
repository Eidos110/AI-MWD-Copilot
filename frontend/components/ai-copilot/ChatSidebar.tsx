'use client';
import { useState, useRef, useEffect } from 'react';
import { useChatStore } from '@/stores/useChatStore';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { useUIStore } from '@/stores/useUIStore';
import { Button } from '@/components/ui/Button';
import { Send, X, Sparkles } from 'lucide-react';
import clsx from 'clsx';

const EXAMPLE_QUERIES = [
  'Show intervals where porosity > 10%',
  'What zones have hydrocarbon potential?',
  'Explain the pressure trend at current depth',
  'Which sensors are most important for porosity?',
];

export function ChatSidebar() {
  const { messages, isTyping, addMessage, setTyping } = useChatStore();
  const { toggleChatSidebar } = useUIStore();
  const { filteredData } = useWellDataStore();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setInput('');
    addMessage({ role: 'user', content: userMsg });

    setTyping(true);
    // Simulate AI response (in production, this would call an LLM endpoint)
    setTimeout(() => {
      const response = generateMockResponse(userMsg, filteredData);
      addMessage({ role: 'assistant', content: response });
      setTyping(false);
    }, 1000 + Math.random() * 1500);
  };

  return (
    <aside className="w-80 bg-white border-l border-gray-200 flex flex-col">
      <div className="p-3 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary-500" />
          <h3 className="font-semibold text-sm">AI Copilot</h3>
        </div>
        <Button variant="ghost" size="sm" onClick={toggleChatSidebar}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs text-gray-500 mb-3">Try asking:</p>
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                onClick={() => { setInput(q); }}
                className="w-full text-left text-xs p-2 rounded-lg bg-gray-50 hover:bg-gray-100 text-gray-700 transition"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={clsx('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}
          >
            <div
              className={clsx('max-w-[85%] rounded-lg px-3 py-2 text-sm', {
                'bg-primary-500 text-white': msg.role === 'user',
                'bg-gray-100 text-gray-900': msg.role === 'assistant',
              })}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-3 py-2 text-sm text-gray-500">
              <span className="animate-pulse">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-3 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask the AI..."
            className="flex-1 text-sm border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
          <Button size="sm" onClick={handleSend} disabled={!input.trim() || isTyping}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </aside>
  );
}

function generateMockResponse(query: string, data: any[]): string {
  const q = query.toLowerCase();
  if (q.includes('porosity') && q.includes('10')) {
    const highPoro = data.filter((d) => Number(d['PHI_COMBINED']) > 0.1);
    return `Found ${highPoro.length} depth points with porosity > 10%. Range: ${highPoro[0]?.DEPTH?.toFixed(0) || 'N/A'}m - ${highPoro[highPoro.length - 1]?.DEPTH?.toFixed(0) || 'N/A'}m. These intervals show good reservoir potential.`;
  }
  if (q.includes('hydrocarbon') || q.includes('pay zone')) {
    const payZones = data.filter((d) => d['FLUID_CLASS'] === 'Pay Zone' || d['FLUID_CLASS'] === 'Potential Reservoir');
    return `Identified ${payZones.length} data points classified as hydrocarbon-bearing zones. The model flags these based on elevated resistivity (RT >= 20 ohm.m) and gas readings (gas >= 10).`;
  }
  if (q.includes('pressure')) {
    return `Pressure trends show a normal compaction gradient in the upper section, transitioning to slight overpressure below ~${Math.floor(data.length * 0.6 + 2000)}m. The corrected drilling exponent anomaly correlates with the pressure increase. Monitor mud weight carefully in this interval.`;
  }
  if (q.includes('sensor') || q.includes('important')) {
    return `Based on SHAP analysis, the top features for porosity prediction are: 1) Bulk Density (strongest), 2) Neutron Porosity, 3) Gamma Ray. Missing any of these sensors significantly degrades model accuracy. Resistivity is critical for fluid classification.`;
  }
  return `I analyzed ${data.length} depth points in the current view. The data shows typical well-logging signatures for this formation. Would you like me to focus on a specific property (porosity, fluid, pressure) or depth interval?`;
}
