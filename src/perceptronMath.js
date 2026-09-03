/**
 * Perceptron Math Utilities
 * Handles all calculations, training logic, and predictions
 */

export const DATASET = [
  { id: 1, temperature: 22, humidity: 45, label: 1 },
  { id: 2, temperature: 24, humidity: 50, label: 1 },
  { id: 3, temperature: 26, humidity: 55, label: 1 },
  { id: 4, temperature: 28, humidity: 60, label: 1 },
  { id: 5, temperature: 30, humidity: 65, label: 1 },
  { id: 6, temperature: 32, humidity: 70, label: 0 },
  { id: 7, temperature: 20, humidity: 75, label: 0 },
  { id: 8, temperature: 18, humidity: 80, label: 0 },
  { id: 9, temperature: 34, humidity: 85, label: 0 },
  { id: 10, temperature: 36, humidity: 90, label: 0 },
];

// ═══════════════════════════════════════════════════════════════
// ACTIVATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════

export const sigmoid = (z) => {
  const clamped = Math.max(-60, Math.min(60, z));
  return 1 / (1 + Math.exp(-clamped));
};

export const stepFn = (z) => (z >= 0 ? 1 : 0);

// ═══════════════════════════════════════════════════════════════
// CORE CALCULATIONS
// ═══════════════════════════════════════════════════════════════

/**
 * Normalize input values to 0-1 range for better perceptron performance
 */
export const normalize = (value, min, max) => {
  if (max === min) return 0;
  return (value - min) / (max - min);
};

/**
 * Calculate weighted sum (z = w1*x1 + w2*x2 + b)
 */
export const calcZ = (temp, humidity, weights, bias) => {
  return weights[0] * temp + weights[1] * humidity + bias;
};

/**
 * Make a prediction using step function
 */
export const predict = (temp, humidity, weights, bias) => {
  const z = calcZ(temp, humidity, weights, bias);
  return stepFn(z);
};

/**
 * Get prediction with confidence (using sigmoid)
 */
export const predictWithConfidence = (temp, humidity, weights, bias) => {
  const z = calcZ(temp, humidity, weights, bias);
  const confidence = sigmoid(z);
  return {
    prediction: stepFn(z),
    confidence: confidence,
    z: z,
  };
};

/**
 * Clamp value between bounds
 */
export const clamp = (value, min, max) => {
  return Math.min(Math.max(value, min), max);
};

// ═══════════════════════════════════════════════════════════════
// TRAINING
// ═══════════════════════════════════════════════════════════════

/**
 * Single training step on one sample
 * Returns updated weights, bias, and error info
 */
export function trainStep(sample, weights, bias, learningRate) {
  const pred = predict(sample.temperature, sample.humidity, weights, bias);
  const error = sample.label - pred;

  // Only update if there was an error
  const newWeights = [
    clamp(weights[0] + learningRate * error * sample.temperature, -10, 10),
    clamp(weights[1] + learningRate * error * sample.humidity, -10, 10),
  ];

  const newBias = clamp(bias + learningRate * error, -10, 10);

  return {
    weights: newWeights,
    bias: newBias,
    prediction: pred,
    error: error,
    updated: error !== 0,
    z: calcZ(sample.temperature, sample.humidity, weights, bias),
  };
}

/**
 * Calculate accuracy on entire dataset
 */
export const calculateAccuracy = (weights, bias, dataset = DATASET) => {
  const correct = dataset.filter(
    (sample) =>
      predict(sample.temperature, sample.humidity, weights, bias) === sample.label
  ).length;
  return (correct / dataset.length) * 100;
};

/**
 * Get predictions for all samples in dataset
 */
export const getPredictionsForDataset = (weights, bias, dataset = DATASET) => {
  return dataset.map((sample) => ({
    ...sample,
    prediction: predict(sample.temperature, sample.humidity, weights, bias),
    z: calcZ(sample.temperature, sample.humidity, weights, bias),
    correct: predict(sample.temperature, sample.humidity, weights, bias) === sample.label,
  }));
};

// ═══════════════════════════════════════════════════════════════
// DECISION BOUNDARY
// ═══════════════════════════════════════════════════════════════

/**
 * Get decision boundary line points
 * Boundary: w[0]*T + w[1]*H + b = 0
 */
export function getBoundaryLine(weights, bias) {
  if (Math.abs(weights[1]) < 1e-6) {
    if (Math.abs(weights[0]) < 1e-6) return null;
    const T = -bias / weights[0];
    return { type: "vertical", T };
  }

  const hAt = (t) => -(weights[0] * t + bias) / weights[1];
  const pts = [];

  const add = (t, h) => {
    if (t >= 15 && t <= 40 && h >= 30 && h <= 100) {
      pts.push({ t, h });
    }
  };

  if (Math.abs(weights[0]) > 1e-6) {
    add(-(weights[1] * 100 + bias) / weights[0], 100);
    add(-(weights[1] * 30 + bias) / weights[0], 30);
  }

  add(15, hAt(15));
  add(40, hAt(40));

  if (pts.length < 2) return null;
  return { type: "line", p1: pts[0], p2: pts[1] };
}

// ═══════════════════════════════════════════════════════════════
// PERSISTENCE
// ═══════════════════════════════════════════════════════════════

export const STORAGE_KEY = "perceptron_predictions";

/**
 * Save predictions to localStorage
 */
export function savePrediction(temp, humidity, prediction, confidence) {
  try {
    const existing = getPredictionHistory();
    const newPrediction = {
      id: Date.now(),
      temperature: temp,
      humidity: humidity,
      prediction: prediction,
      confidence: parseFloat(confidence.toFixed(4)),
      timestamp: new Date().toISOString(),
    };
    const updated = [...existing, newPrediction];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return newPrediction;
  } catch (e) {
    console.error("Failed to save prediction:", e);
    return null;
  }
}

/**
 * Get all saved predictions from localStorage
 */
export function getPredictionHistory() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (e) {
    console.error("Failed to load predictions:", e);
    return [];
  }
}

/**
 * Clear all predictions from localStorage
 */
export function clearPredictionHistory() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    return true;
  } catch (e) {
    console.error("Failed to clear predictions:", e);
    return false;
  }
}

/**
 * Delete a single prediction by ID
 */
export function deletePrediction(id) {
  try {
    const existing = getPredictionHistory();
    const updated = existing.filter((p) => p.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return true;
  } catch (e) {
    console.error("Failed to delete prediction:", e);
    return false;
  }
}
