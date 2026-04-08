const ctx = self as unknown as Worker;

ctx.onmessage = async (e: MessageEvent) => {
  const { type, payload, id } = e.data;

  try {
    switch (type) {
      case 'parseCSV': {
        const { content, options } = payload;
        const result = await parseCSVInWorker(content, options);
        ctx.postMessage({ id, success: true, result });
        break;
      }
      case 'parseExcel': {
        const { buffer } = payload;
        const result = await parseExcelInWorker(buffer);
        ctx.postMessage({ id, success: true, result });
        break;
      }
      default:
        ctx.postMessage({ id, success: false, error: 'Unknown message type' });
    }
  } catch (error) {
    ctx.postMessage({ id, success: false, error: (error as Error).message });
  }
};

interface ParseOptions {
  header?: boolean;
  skipEmptyRows?: boolean;
  columnTypes?: Record<string, 'number' | 'string' | 'date'>;
}

async function parseCSVInWorker(
  content: string,
  options: ParseOptions = {}
): Promise<{ headers: string[]; rows: Record<string, unknown>[]; rowCount: number }> {
  const lines = content.split(/\r?\n/).filter((line) => line.trim());
  
  if (lines.length === 0) {
    return { headers: [], rows: [], rowCount: 0 };
  }

  const headers = parseCSVLine(lines[0]);
  const rows: Record<string, unknown>[] = [];
  const startIndex = options.header !== false ? 1 : 0;
  const skipEmpty = options.skipEmptyRows !== false;

  for (let i = startIndex; i < lines.length; i++) {
    if (skipEmpty && !lines[i].trim()) continue;

    const values = parseCSVLine(lines[i]);
    const row: Record<string, unknown> = {};

    for (let j = 0; j < headers.length; j++) {
      let value = values[j] || '';
      const colType = options.columnTypes?.[headers[j]];

      if (colType === 'number') {
        const num = parseFloat(value);
        row[headers[j]] = isNaN(num) ? null : num;
      } else if (colType === 'date') {
        const date = new Date(value);
        row[headers[j]] = isNaN(date.getTime()) ? null : date.toISOString();
      } else {
        row[headers[j]] = value;
      }
    }

    rows.push(row);
  }

  return { headers, rows, rowCount: rows.length };
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const nextChar = line[i + 1];

    if (inQuotes) {
      if (char === '"' && nextChar === '"') {
        current += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        current += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ',') {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
  }

  result.push(current.trim());
  return result;
}

async function parseExcelInWorker(
  buffer: ArrayBuffer
): Promise<{ headers: string[]; rows: Record<string, unknown>[]; rowCount: number }> {
  // Dynamic import for XLSX to keep worker bundle small
  const XLSX = await import('xlsx');
  
  const workbook = XLSX.read(buffer, { type: 'array' });
  const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
  const jsonData = XLSX.utils.sheet_to_json(firstSheet, { defval: null });

  if (jsonData.length === 0) {
    return { headers: [], rows: [], rowCount: 0 };
  }

  const headers = Object.keys(jsonData[0] as Record<string, unknown>);
  const rows = jsonData.map((row) => {
    const processedRow: Record<string, unknown> = {};
    for (const key of headers) {
      const val = (row as Record<string, unknown>)[key];
      if (typeof val === 'number' && !isNaN(val)) {
        processedRow[key] = val;
      } else if (val instanceof Date) {
        processedRow[key] = val.toISOString();
      } else {
        processedRow[key] = val;
      }
    }
    return processedRow;
  });

  return { headers, rows, rowCount: rows.length };
}

export {};