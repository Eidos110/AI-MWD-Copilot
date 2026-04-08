export function downsampleLTTB(data: number[][], threshold: number): number[][] {
  if (data.length <= threshold) return data;
  
  const sampled: number[][] = [data[0]];
  const bucketSize = (data.length - 2) / (threshold - 2);
  let a = 0;
  
  for (let i = 0; i < threshold - 2; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.floor((i + 2) * bucketSize) + 1;
    const nextBucketStart = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length - 1);
    const nextBucketEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, data.length);
    
    let avgX = 0, avgY = 0, count = 0;
    for (let j = nextBucketStart; j < nextBucketEnd; j++) {
      avgX += data[j][0];
      avgY += data[j][1];
      count++;
    }
    avgX /= count;
    avgY /= count;
    
    let maxArea = -1;
    let maxIdx = bucketStart;
    const pointA = data[a];
    
    for (let j = bucketStart; j < bucketEnd && j < data.length; j++) {
      const area = Math.abs(
        (pointA[0] - avgX) * (data[j][1] - pointA[1]) -
        (pointA[0] - data[j][0]) * (avgY - pointA[1])
      );
      if (area > maxArea) {
        maxArea = area;
        maxIdx = j;
      }
    }
    
    sampled.push(data[maxIdx]);
    a = maxIdx;
  }
  
  sampled.push(data[data.length - 1]);
  return sampled;
}

export function downsampleForChart(
  depths: number[],
  values: number[],
  maxPoints = 3000
): { depths: number[]; values: number[] } {
  if (depths.length <= maxPoints) return { depths, values };
  const paired = depths.map((d, i) => [d, values[i]]);
  const downsampled = downsampleLTTB(paired, maxPoints);
  return {
    depths: downsampled.map((p) => p[0]),
    values: downsampled.map((p) => p[1]),
  };
}
