'use client';
import { ChatMessage as ChatMessageType } from '@/types';
import { User, Bot } from 'lucide-react';
import clsx from 'clsx';
import { format } from '@/lib/utils/date';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={clsx('flex gap-2', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="flex-shrink-0 mt-1">
          <Bot className="h-5 w-5 text-primary-500" />
        </div>
      )}
      <div
        className={clsx('max-w-[85%] rounded-lg px-3 py-2 text-sm', {
          'bg-primary-500 text-white': isUser,
          'bg-gray-100 text-gray-900': !isUser,
        })}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
        {message.highlightedDepths && message.highlightedDepths.length > 0 && (
          <div className="mt-2 pt-2 border-t border-opacity-20 border-current text-xs opacity-75">
            <span>Depths: </span>
            {message.highlightedDepths.slice(0, 5).map((d, i) => (
              <span key={i} className="inline-block bg-white bg-opacity-20 rounded px-1 mr-1">
                {d.toFixed(0)}m
              </span>
            ))}
            {message.highlightedDepths.length > 5 && (
              <span className="text-xs">+{message.highlightedDepths.length - 5} more</span>
            )}
          </div>
        )}
        <div className={clsx('text-xs mt-1', isUser ? 'text-primary-100' : 'text-gray-400')}>
          {format(message.timestamp)}
        </div>
      </div>
      {isUser && (
        <div className="flex-shrink-0 mt-1">
          <User className="h-5 w-5 text-primary-500" />
        </div>
      )}
    </div>
  );
}
