'use client';

import { useEffect } from 'react';
import { useInterpretationStore } from '@/stores/useInterpretationStore';
import { usePredictionStore } from '@/stores/usePredictionStore';
import { useWellDataStore } from '@/stores/useWellDataStore';
import { ZoneSummaryPanel, ZoneRecommendations } from '@/components/interpret/ZoneSummary';
import { PetrophysicsPanel } from '@/components/interpret/PetrophysicsPanel';
import { DrillingDecisionPanel, CriticalDepthsTable } from '@/components/interpret/DrillingDecisionPanel';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { RefreshCw, Play } from 'lucide-react';

export function InterpretTab() {
  const { fetchInterpretation, fetchZoneInterpretation, fetchPetrophysics, fetchDrilling, zones, petrophysics, drilling, loading, error } = useInterpretationStore();
  const { predictions, isLoading: predLoading } = usePredictionStore();
  const { filteredData, depthRange } = useWellDataStore();

  // Check if predictions have actual data (not just empty object)
  const hasPredictions = (predictions?.porosity?.predictions?.length ?? 0) > 0 
    || (predictions?.fluid?.predictions?.length ?? 0) > 0 
    || (predictions?.pressure?.predictions?.length ?? 0) > 0;

  useEffect(() => {
    // Auto-run interpretation when predictions are available
    if (filteredData.length > 0 && hasPredictions) {
      const dataWithPredictions = filteredData.map((row, idx) => {
        const newRow = { ...row };
        if (predictions?.porosity?.predictions?.[idx] !== undefined) {
          newRow.porosity = predictions.porosity.predictions[idx];
        }
        if (predictions?.pressure?.predictions?.[idx] !== undefined) {
          newRow.pressure = predictions.pressure.predictions[idx];
        }
        if (predictions?.fluid?.predictions?.[idx] !== undefined) {
          newRow.fluid = predictions.fluid.predictions[idx];
        }
        if (predictions?.fluid?.confidence?.[idx] !== undefined) {
          newRow.confidence = predictions.fluid.confidence[idx];
        }
        return newRow;
      });
      fetchInterpretation(dataWithPredictions, depthRange);
    }
  }, [hasPredictions, filteredData.length]);

  const hasData = zones || petrophysics || drilling;

  const handleRunPredictions = () => {
    window.dispatchEvent(new CustomEvent('run-predictions'));
  };

  const handleRunInterpretation = () => {
    if (filteredData.length === 0) return;
    
    // Build data with predictions - only include predictions that exist
    const dataWithPredictions = filteredData.map((row, idx) => {
      const newRow = { ...row };
      
      // Add predictions if they exist
      if (predictions?.porosity?.predictions?.[idx] !== undefined) {
        newRow.porosity = predictions.porosity.predictions[idx];
      }
      if (predictions?.pressure?.predictions?.[idx] !== undefined) {
        newRow.pressure = predictions.pressure.predictions[idx];
      }
      if (predictions?.fluid?.predictions?.[idx] !== undefined) {
        newRow.fluid = predictions.fluid.predictions[idx];
      }
      if (predictions?.fluid?.confidence?.[idx] !== undefined) {
        newRow.confidence = predictions.fluid.confidence[idx];
      }
      
      return newRow;
    });
    
    console.log('[InterpretTab] Running interpretation with data:', dataWithPredictions.slice(0, 2));
    fetchInterpretation(dataWithPredictions, depthRange);
  };

  const handleRunIndividual = (type: 'zones' | 'petrophysics' | 'drilling') => {
    if (filteredData.length === 0) return;
    const dataWithPredictions = filteredData.map((row, idx) => {
      const newRow = { ...row };
      if (predictions?.porosity?.predictions?.[idx] !== undefined) {
        newRow.porosity = predictions.porosity.predictions[idx];
      }
      if (predictions?.pressure?.predictions?.[idx] !== undefined) {
        newRow.pressure = predictions.pressure.predictions[idx];
      }
      if (predictions?.fluid?.predictions?.[idx] !== undefined) {
        newRow.fluid = predictions.fluid.predictions[idx];
      }
      if (predictions?.fluid?.confidence?.[idx] !== undefined) {
        newRow.confidence = predictions.fluid.confidence[idx];
      }
      return newRow;
    });
    if (type === 'zones') fetchZoneInterpretation(dataWithPredictions, depthRange);
    if (type === 'petrophysics') fetchPetrophysics(dataWithPredictions, depthRange);
    if (type === 'drilling') fetchDrilling(dataWithPredictions, depthRange);
  };

  // Debug info
  console.log('[InterpretTab] filteredData:', filteredData.length, 'hasPredictions:', hasPredictions, 'predictions:', predictions);
  console.log('[InterpretTab] zones:', zones, 'petrophysics:', petrophysics, 'drilling:', drilling, 'hasData:', hasData, 'error:', error);

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Interpretation & Decision Support</h2>
          <p className="text-sm text-gray-500">AI-powered analysis for drilling decisions</p>
        </div>
        <div className="flex gap-2">
          {/* Always show Run Predictions button when no predictions or predictions incomplete */}
          {(!hasPredictions || predLoading) && (
            <Button variant="primary" size="sm" onClick={handleRunPredictions} disabled={predLoading || filteredData.length === 0}>
              <Play className="h-4 w-4 mr-1" />
              {predLoading ? 'Running...' : 'Run Predictions'}
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={handleRunInterpretation} disabled={loading || filteredData.length === 0}>
            <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Loading...' : 'Run Interpretation'}
          </Button>
        </div>
      </div>

      {filteredData.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-gray-500">Please load data from the Dashboard tab first.</p>
          </CardContent>
        </Card>
      )}

      {/* Loading state */}
      {(loading || predLoading) && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-blue-500 animate-pulse">
              {predLoading ? 'Running predictions...' : 'Running interpretation...'}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Error state - show prominently */}
      {error && (
        <Card>
          <CardContent className="p-4 bg-red-50 border border-red-200 rounded">
            <p className="text-red-600 font-semibold">Error: {error}</p>
          </CardContent>
        </Card>
      )}

      {/* No predictions yet */}
      {filteredData.length > 0 && !hasPredictions && !predLoading && !loading && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-gray-500">Click "Run Predictions" to generate predictions first, then run interpretation.</p>
          </CardContent>
        </Card>
      )}

      {/* No interpretation yet but predictions exist */}
      {filteredData.length > 0 && hasPredictions && !hasData && !loading && (
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-gray-500">Click "Run Interpretation" to analyze the predictions.</p>
          </CardContent>
        </Card>
      )}

      {hasData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <ZoneSummaryPanel />
          <PetrophysicsPanel />
          <DrillingDecisionPanel />
        </div>
      )}

      {hasData && zones?.zones && zones.zones.length > 0 && (
        <ZoneRecommendations />
      )}

      {hasData && drilling?.critical_depths && drilling.critical_depths.length > 0 && (
        <CriticalDepthsTable />
      )}
    </div>
  );
}