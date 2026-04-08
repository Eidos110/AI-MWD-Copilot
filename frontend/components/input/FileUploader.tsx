'use client';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadFile } from '@/lib/api';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { Button } from '@/components/ui/Button';
import { notify } from '@/components/ui/Toast';
import { Upload, FileSpreadsheet, X, CheckCircle } from 'lucide-react';

interface FileUploaderProps {
  onUploadComplete?: (filename: string, rows: number) => void;
  className?: string;
}

export function FileUploader({ onUploadComplete, className }: FileUploaderProps) {
  const { rawData, setData, setUploadedFileName, setDepthRange, setLoading } = useWellDataStore();
  const [uploadedFile, setUploadedFile] = useState<{ name: string; rows: number } | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    setLoading(true);
    try {
      const result = await uploadFile(file);
      setData(
        result.preview.concat(
          rawData.length > result.preview.length ? rawData.slice(result.preview.length) : []
        ),
        result.columns
      );
      setUploadedFileName(result.filename);
      setUploadedFile({ name: result.filename, rows: result.rows });
      if (result.depth_range) {
        setDepthRange(result.depth_range);
      }
      notify.success(`Loaded ${result.rows.toLocaleString()} rows from ${result.filename}`);
      onUploadComplete?.(result.filename, result.rows);
    } catch (err: any) {
      notify.error(`Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [rawData, setData, setUploadedFileName, setDepthRange, setLoading, onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
    maxSize: 200 * 1024 * 1024,
  });

  const clearUpload = () => {
    setUploadedFile(null);
    setUploadedFileName(null);
  };

  return (
    <div className={className}>
      <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
        <Upload className="h-4 w-4" /> Upload Data
      </h3>

      {uploadedFile ? (
        <div className="border border-green-200 bg-green-50 rounded-lg p-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-green-800 truncate">{uploadedFile.name}</p>
                <p className="text-xs text-green-600">{uploadedFile.rows.toLocaleString()} rows loaded</p>
              </div>
            </div>
            <button
              onClick={clearUpload}
              className="text-gray-400 hover:text-gray-600 p-0.5"
              aria-label="Clear upload"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            isDragActive
              ? isDragReject
                ? 'border-red-400 bg-red-50'
                : 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} />
          <FileSpreadsheet className="h-8 w-8 mx-auto mb-2 text-gray-400" />
          <p className="text-sm text-gray-600">
            {isDragActive
              ? isDragReject
                ? 'Unsupported file type'
                : 'Drop file here...'
              : 'Drag & drop CSV or Excel file'}
          </p>
          <p className="text-xs text-gray-400 mt-1">or click to browse (max 200MB)</p>
        </div>
      )}
    </div>
  );
}
