import { useEffect, useRef, useState, useCallback } from 'react';
import { WellDataRow } from '@/types';

interface ParseResult {
  headers: string[];
  rows: Record<string, unknown>[];
  rowCount: number;
}

interface DownsampleResult {
  depths: number[];
  values: number[];
}

type WorkerMessage = {
  id: number;
  type: string;
  payload?: any;
};

type WorkerResponse = {
  id: number;
  success: boolean;
  result?: any;
  error?: string;
};

export function useParseWorker() {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef<Map<number, { resolve: Function; reject: Function }>>(new Map());
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    workerRef.current = new Worker(
      new URL('./workers/parseWorker.ts', import.meta.url),
      { type: 'module' }
    );

    workerRef.current.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const { id, success, result, error } = e.data;
      const pending = pendingRef.current.get(id);
      if (pending) {
        if (success) {
          pending.resolve(result);
        } else {
          pending.reject(new Error(error));
        }
        pendingRef.current.delete(id);
      }
    };

    setIsReady(true);

    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  const sendMessage = useCallback((type: string, payload?: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = Date.now() + Math.random();
      pendingRef.current.set(id, { resolve, reject });
      workerRef.current.postMessage({ type, payload, id } as WorkerMessage);
    });
  }, []);

  const parseCSV = useCallback((content: string, options?: any) => {
    return sendMessage('parseCSV', { content, options }) as Promise<ParseResult>;
  }, [sendMessage]);

  const parseExcel = useCallback((buffer: ArrayBuffer) => {
    return sendMessage('parseExcel', { buffer }) as Promise<ParseResult>;
  }, [sendMessage]);

  return { isReady, parseCSV, parseExcel };
}

export function useDownsampleWorker() {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef<Map<number, { resolve: Function; reject: Function }>>(new Map());
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    workerRef.current = new Worker(
      new URL('./workers/downsampleWorker.ts', import.meta.url),
      { type: 'module' }
    );

    workerRef.current.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const { id, success, result, error } = e.data;
      const pending = pendingRef.current.get(id);
      if (pending) {
        if (success) {
          pending.resolve(result);
        } else {
          pending.reject(new Error(error));
        }
        pendingRef.current.delete(id);
      }
    };

    setIsReady(true);

    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  const sendMessage = useCallback((type: string, payload?: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = Date.now() + Math.random();
      pendingRef.current.set(id, { resolve, reject });
      workerRef.current.postMessage({ type, payload, id } as WorkerMessage);
    });
  }, []);

  const downsample = useCallback(
    (depths: number[], values: number[], maxPoints = 3000) => {
      return sendMessage('downsampleLTTB', { depths, values, maxPoints }) as Promise<DownsampleResult>;
    },
    [sendMessage]
  );

  const downsampleMultiple = useCallback(
    (datasets: DownsampleResult[], maxPoints = 3000) => {
      return sendMessage('downsampleMultiple', { datasets, maxPoints }) as Promise<DownsampleResult[]>;
    },
    [sendMessage]
  );

  const batchDownsample = useCallback(
    (requests: { depths: number[]; values: number[]; maxPoints: number }[]) => {
      return sendMessage('batchDownsample', { requests }) as Promise<DownsampleResult[]>;
    },
    [sendMessage]
  );

  return { isReady, downsample, downsampleMultiple, batchDownsample };
}

export function useDataProcessor() {
  const parseWorker = useParseWorker();
  const downsampleWorker = useDownsampleWorker();

  const processCSV = useCallback(
    async (content: string, options?: any): Promise<WellDataRow[]> => {
      if (!parseWorker.isReady) {
        throw new Error('Parse worker not ready');
      }

      const result = await parseWorker.parseCSV(content, options);
      return result.rows as WellDataRow[];
    },
    [parseWorker]
  );

  const processExcel = useCallback(
    async (buffer: ArrayBuffer): Promise<WellDataRow[]> => {
      if (!parseWorker.isReady) {
        throw new Error('Parse worker not ready');
      }

      const result = await parseWorker.parseExcel(buffer);
      return result.rows as WellDataRow[];
    },
    [parseWorker]
  );

  const processWithDownsample = useCallback(
    async (
      data: WellDataRow[],
      columns: string[],
      maxPoints = 3000
    ): Promise<Map<string, DownsampleResult>> => {
      if (!downsampleWorker.isReady) {
        throw new Error('Downsample worker not ready');
      }

      const requests = columns.map((col) => {
        const depths = data.map((d) => d.DEPTH);
        const values = data.map((d) => Number(d[col]) || 0);
        return { depths, values, maxPoints };
      });

      const results = await downsampleWorker.batchDownsample(requests);
      
      const resultMap = new Map<string, DownsampleResult>();
      columns.forEach((col, i) => {
        resultMap.set(col, results[i]);
      });

      return resultMap;
    },
    [downsampleWorker]
  );

  return {
    isReady: parseWorker.isReady && downsampleWorker.isReady,
    processCSV,
    processExcel,
    processWithDownsample,
  };
}