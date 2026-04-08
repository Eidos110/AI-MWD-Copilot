const ctx = self as unknown as Worker;

ctx.onmessage = async (e: MessageEvent) => {
  const { type, payload, id } = e.data;

  try {
    switch (type) {
      case 'downsampleLTTB': {
        const { depths, values, maxPoints } = payload;
        const result = downsampleLTTBWorker(depths, values, maxPoints);
        ctx.postMessage({ id, success: true, result });
        break;
      }
      case 'downsampleMultiple': {
        const { datasets, maxPoints } = payload;
        const result = datasets.map(
          (d: { depths: number[]; values: number[] }) =>
            downsampleLTTBWorker(d.depths, d.values, maxPoints)
        );
        ctx.postMessage({ id, success: true, result });
        break;
      }
      case 'batchDownsample': {
        const { requests } = payload;
        const results = requests.map(
          (r: { depths: number[]; values: number[]; maxPoints: number }) =>
            downsampleLTTBWorker(r.depths, r.values, r.maxPoints)
        );
        ctx.postMessage({ id, success: true, result: results });
        break;
      }
      default:
        ctx.postMessage({ id, success: false, error: 'Unknown message type' });
    }
  } catch (error) {
    ctx.postMessage({ id, success: false, error: (error as Error).message });
  }
};

function downsampleLTTBWorker(
  depths: number[],
  values: number[],
  maxPoints: number
): { depths: number[]; values: number[] } {
  if (depths.length <= maxPoints || maxPoints < 3) {
    return { depths, values };
  }

  const n = depths.length;
  const sampled: { depth: number; value: number }[] = [];

  sampled.push({ depth: depths[0], value: values[0] });

  const bucketSize = (n - 2) / (maxPoints - 2);
  let prevIndex = 0;

  for (let i = 0; i < maxPoints - 2; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, n - 1);

    let avgX = 0;
    let avgY = 0;
    const nextBucketStart = Math.min(Math.floor((i + 2) * bucketSize) + 1, n - 1);
    const nextBucketEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, n);
    const nextBucketSize = nextBucketEnd - nextBucketStart;

    for (let j = nextBucketStart; j < nextBucketEnd; j++) {
      avgX += depths[j];
      avgY += values[j];
    }

    if (nextBucketSize > 0) {
      avgX /= nextBucketSize;
      avgY /= nextBucketSize;
    } else {
      avgX = depths[bucketEnd];
      avgY = values[bucketEnd];
    }

    const pointA = { x: depths[prevIndex], y: values[prevIndex] };
    let maxArea = -1;
    let maxIdx = bucketStart;

    for (let j = bucketStart; j < bucketEnd; j++) {
      const area = Math.abs(
        (pointA.x - avgX) * (values[j] - pointA.y) -
        (pointA.x - depths[j]) * (avgY - pointA.y)
      );

      if (area > maxArea) {
        maxArea = area;
        maxIdx = j;
      }
    }

    sampled.push({ depth: depths[maxIdx], value: values[maxIdx] });
    prevIndex = maxIdx;
  }

  sampled.push({ depth: depths[n - 1], value: values[n - 1] });

  return {
    depths: sampled.map((p) => p.depth),
    values: sampled.map((p) => p.value),
  };
}

export {};