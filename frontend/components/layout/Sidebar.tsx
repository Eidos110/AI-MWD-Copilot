'use client';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { useUIStore } from '@/stores/useUIStore';
import { Button } from '@/components/ui/Button';
import { Toggle } from '@/components/ui/Toggle';
import { Slider } from '@/components/ui/Slider';
import { DEPTH_PRESETS } from '@/lib/constants';
import { uploadFile } from '@/lib/api';
import { Upload, MapPin } from 'lucide-react';
import { notify } from '@/components/ui/Toast';

export function Sidebar() {
  const { depthRange, setDepthRange, rawData, setData, setUploadedFileName, setLoading } = useWellDataStore();
  const { showConfidence, setShowConfidence } = useUIStore();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    setLoading(true);
    try {
      const result = await uploadFile(acceptedFiles[0]);
      setData(result.preview.concat(
        rawData.length > result.preview.length ? rawData.slice(result.preview.length) : []
      ), result.columns);
      setUploadedFileName(result.filename);
      if (result.depth_range) {
        setDepthRange(result.depth_range);
      }
      notify.success(`Loaded ${result.rows} rows from ${result.filename}`);
    } catch (err: any) {
      notify.error(`Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [rawData]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] },
    maxFiles: 1,
    maxSize: 200 * 1024 * 1024,
  });

  const dataMin = rawData.length > 0 ? Math.floor(rawData[0].DEPTH) : 0;
  const dataMax = rawData.length > 0 ? Math.ceil(rawData[rawData.length - 1].DEPTH) : 5000;

  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col overflow-y-auto">
      <div className="p-4 space-y-4">
        {/* File Upload */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
            <Upload className="h-4 w-4" /> Upload Data
          </h3>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <p className="text-xs text-gray-500">
              {isDragActive ? 'Drop file here...' : 'Drag & drop CSV/Excel, or click'}
            </p>
          </div>
        </div>

        {/* Depth Presets */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
            <MapPin className="h-4 w-4" /> Depth Presets
          </h3>
          <div className="grid grid-cols-3 gap-1">
            {DEPTH_PRESETS.map((p) => (
              <Button
                key={p.label}
                variant={depthRange.min === p.min && depthRange.max === p.max ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => setDepthRange({ min: p.min, max: p.max })}
                className="text-xs"
              >
                {p.label}
              </Button>
            ))}
          </div>
        </div>

        {/* Custom Depth Range */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Custom Range</h3>
          <div className="space-y-3">
            <Slider
              min={dataMin}
              max={dataMax}
              value={depthRange.min}
              onChange={(v) => setDepthRange({ ...depthRange, min: Math.min(v, depthRange.max - 1) })}
              label="Min Depth (m)"
            />
            <Slider
              min={dataMin}
              max={dataMax}
              value={depthRange.max}
              onChange={(v) => setDepthRange({ ...depthRange, max: Math.max(v, depthRange.min + 1) })}
              label="Max Depth (m)"
            />
          </div>
        </div>

        {/* Settings */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Settings</h3>
          <div className="space-y-2">
            <Toggle enabled={showConfidence} onChange={setShowConfidence} label="Show Confidence" size="sm" />
          </div>
        </div>

        {/* Data Info */}
        {rawData.length > 0 && (
          <div className="text-xs text-gray-500 space-y-1 pt-2 border-t">
            <p>Total: {rawData.length.toLocaleString()} points</p>
            <p>Range: {depthRange.min}m - {depthRange.max}m</p>
            <p>Visible: {(depthRange.max - depthRange.min).toLocaleString()}m</p>
          </div>
        )}
      </div>
    </aside>
  );
}
