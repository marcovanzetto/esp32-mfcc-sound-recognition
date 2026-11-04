/*
 * Low-Cost MFCC-Based Real-Time Sound Recognition
 * ------------------------------------------------
 * Learns a user-defined sound and then detects it in real time on an ESP32,
 * using MFCC feature matrices and sliding-window Frobenius-distance matching.
 *
 * Hardware : ESP32 dev board + MAX9814 electret microphone amplifier (OUT -> GPIO 15)
 * Library  : arduinoFFT >= 2.0 (Enrique Condes)
 * Usage    : open the Serial Monitor at 115200 baud and send 'L' to start learning.
 *
 * See README.md for a full description of the method.
 */

#include <arduinoFFT.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define MIC_PIN 15                 // Analog pin connected to the microphone
#define NUM_SAMPLES 512            // Samples per frame (FFT size)
#define SAMPLE_DELAY_US 79         // Inter-sample delay; with analogRead() overhead gives ~8 kHz
#define NUM_MEL_FILTERS 20         // Number of triangular Mel filters (M)
#define NUM_MFCCS 6                // MFCCs computed (c0..c5); c0 is discarded -> K = 5
#define NUM_TEST_FRAMES 8          // Frames in the test matrix (Nt)
#define NUM_REF_FRAMES 20          // Frames in the reference matrix (Nr), must be > Nt
#define RMS_THRESHOLD 225          // Minimum frame RMS (ADC units) to process a frame
#define PAUSE_TIME_MS 1000         // Pause after each detection
#define NUM_LEARNING_DISTANCES 50  // Distances collected to set the threshold (~4 s)

// Adaptive threshold selection (learning stage 2)
#define INITIAL_PERCENTILE 0.9f    // Starting percentile of the learning distances
#define PERCENTILE_DECAY 1.2f      // Percentile is divided by this factor at each step...
#define THRESHOLD_RATIO 1.6f       // ...until d[percentile] <= THRESHOLD_RATIO * d[3]
#define MAX_REBUILDS 10            // Stop stage 2 early after this many test-matrix restarts

// Optional serial output
#define LOG_DISTANCES 0            // 1 = print the best distance of every monitored frame
#define ENABLE_EXPERIMENT_LOG 0    // 1 = print the START/STOP summaries used for the measurements

#define NUM_KEPT_MFCCS (NUM_MFCCS - 1)  // c0 (frame energy) is not used

#if NUM_REF_FRAMES <= NUM_TEST_FRAMES
#error "NUM_REF_FRAMES must be greater than NUM_TEST_FRAMES"
#endif

// ============================================================================
// GLOBAL STATE
// ============================================================================

float fftReal[NUM_SAMPLES];  // Frame samples, then FFT magnitude spectrum
float fftImag[NUM_SAMPLES];  // Imaginary part of FFT input

float melEnergy[NUM_MEL_FILTERS];  // Log Mel filterbank energies
float mfcc[NUM_MFCCS];             // MFCCs of the current frame

float refMatrix[NUM_KEPT_MFCCS][NUM_REF_FRAMES];    // Reference MFCC matrix R
float testMatrix[NUM_KEPT_MFCCS][NUM_TEST_FRAMES];  // Test MFCC matrix T (sliding)

int testFrameIdx = 0;  // Number of frames pushed into the current test sequence
float rmsValue = 0;    // RMS of the last acquired frame (before normalization)

bool learning2Active = false;     // Learning stage 2 (threshold estimation) running
bool isMonitoringActive = false;  // Monitoring phase running

float distanceTh = 0.0;  // Detection threshold d_th
float learningDistances[NUM_LEARNING_DISTANCES];
int learning2Count = 0;
int rebuildCount = -1;  // Number of times the test matrix was restarted in stage 2

long monitoringStartTime = 0;
long lastDetectionTime = -1e9;  // Timestamp of the last detection (ms)

// Statistics printed by the experiment log
int countDet = 0;
bool isFirstDet = true;
bool isSecondDet = false;
float firstDetTime = 0;
float secondDetTime = 0;
float minDist = 1e3;
bool stopPending = true;

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

float hzToMel(float hz) {
  return 2595.0 * log10(1.0 + hz / 700.0);
}

float melToHz(float mel) {
  return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Computes MFCCs from an FFT magnitude spectrum.
void computeMfcc(float *spectrum, float samplingFreq, float *out) {
  float melMin = hzToMel(0);
  float melMax = hzToMel(samplingFreq / 2.0);
  float melStep = (melMax - melMin) / (NUM_MEL_FILTERS + 1);

  // Filter edges, equally spaced on the Mel scale and mapped to FFT bins
  int binPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    float hz = melToHz(melMin + i * melStep);
    binPoints[i] = floor(NUM_SAMPLES * hz / samplingFreq);
  }

  // Triangular filterbank + log compression
  for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
    float sum = 0.0;
    for (int k = binPoints[m - 1]; k < binPoints[m]; k++) {
      float w = (k - binPoints[m - 1]) / float(binPoints[m] - binPoints[m - 1]);
      sum += w * spectrum[k];
    }
    for (int k = binPoints[m]; k < binPoints[m + 1]; k++) {
      float w = (binPoints[m + 1] - k) / float(binPoints[m + 1] - binPoints[m]);
      sum += w * spectrum[k];
    }
    melEnergy[m - 1] = log(sum + 1e-12);
  }

  // DCT-II of the log Mel energies
  for (int n = 0; n < NUM_MFCCS; n++) {
    float s = 0.0;
    for (int m = 0; m < NUM_MEL_FILTERS; m++)
      s += melEnergy[m] * cos(M_PI * n * (m + 0.5) / NUM_MEL_FILTERS);
    out[n] = s;
  }
}

// Hamming window + FFT + magnitude + MFCC on the frame stored in fftReal.
void computeFrameMfcc(float samplingFreq) {
  for (int i = 0; i < NUM_SAMPLES; i++) fftImag[i] = 0;

  ArduinoFFT<float> FFT(fftReal, fftImag, NUM_SAMPLES, samplingFreq);
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  computeMfcc(fftReal, samplingFreq, mfcc);
}

// ============================================================================
// MATCHING
// ============================================================================

// Frobenius distance between the test matrix and the reference submatrix
// starting at column startIdx.
float computeFrobenius(int startIdx) {
  float sum = 0.0;
  for (int r = 0; r < NUM_KEPT_MFCCS; r++) {
    for (int c = 0; c < NUM_TEST_FRAMES; c++) {
      float diff = refMatrix[r][startIdx + c] - testMatrix[r][c];
      sum += diff * diff;
    }
  }
  return sqrt(sum);
}

// Minimum distance over all sliding positions of the test matrix along the reference.
float findBestDistance() {
  float bestDist = 1e4;
  for (int start = 0; start <= NUM_REF_FRAMES - NUM_TEST_FRAMES; start++) {
    float d = computeFrobenius(start);
    if (d < bestDist) bestDist = d;
  }
  return bestDist;
}

// Appends the current frame's MFCCs (c1..c5) to the test matrix,
// shifting out the oldest column once the matrix is full.
void updateTestMatrix() {
  int col = testFrameIdx;
  if (testFrameIdx >= NUM_TEST_FRAMES) {
    for (int c = 1; c < NUM_TEST_FRAMES; c++)
      for (int r = 0; r < NUM_KEPT_MFCCS; r++)
        testMatrix[r][c - 1] = testMatrix[r][c];
    col = NUM_TEST_FRAMES - 1;
  }
  for (int r = 0; r < NUM_KEPT_MFCCS; r++) testMatrix[r][col] = mfcc[r + 1];
}

// ============================================================================
// ACQUISITION & PREPROCESSING
// ============================================================================

// Acquires one frame into buffer, removes DC, high-pass filters it and
// normalizes it by its RMS. Sets rmsValue and returns the measured sampling rate.
void acquireAndProcessFrame(float *buffer, float &samplingFreq) {
  unsigned long t0 = micros();

  float sum = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    buffer[i] = analogRead(MIC_PIN);
    sum += buffer[i];
    delayMicroseconds(SAMPLE_DELAY_US);
  }

  unsigned long t1 = micros();
  samplingFreq = (float)NUM_SAMPLES / ((t1 - t0) / 1e6);

  // DC-offset removal
  float mean = sum / NUM_SAMPLES;
  for (int i = 0; i < NUM_SAMPLES; i++) buffer[i] -= mean;

  // First-order high-pass IIR filter
  const float alpha = 0.98;
  float prevX = buffer[0];
  float prevY = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    float y = alpha * (prevY + buffer[i] - prevX);
    prevX = buffer[i];
    buffer[i] = y;
    prevY = y;
  }

  // RMS of the filtered frame
  float sumSq = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) sumSq += buffer[i] * buffer[i];
  float rmsFrame = sqrt(sumSq / NUM_SAMPLES);
  if (rmsFrame < 1e-6) rmsFrame = 1e-6;

  // Amplitude normalization
  for (int i = 0; i < NUM_SAMPLES; i++) buffer[i] /= rmsFrame;

  rmsValue = rmsFrame;
}

// RMS gate applied at the start of a test sequence and then every
// NUM_TEST_FRAMES frames. Returns false if the frame must be discarded
// (in which case the test sequence is reset). Sets *newSequence when a
// fresh test sequence starts with this frame.
bool passesRmsGate(bool *newSequence) {
  *newSequence = false;
  if (testFrameIdx != 0 && testFrameIdx != NUM_TEST_FRAMES && testFrameIdx != 2 * NUM_TEST_FRAMES)
    return true;

  if (rmsValue < RMS_THRESHOLD) {
    testFrameIdx = 0;
    return false;
  }

  if (testFrameIdx == 0) *newSequence = true;
  else testFrameIdx = NUM_TEST_FRAMES;  // keep the counter bounded
  return true;
}

// ============================================================================
// LEARNING
// ============================================================================

// Stage 1: records NUM_REF_FRAMES consecutive frames into the reference matrix,
// starting from the first frame whose RMS exceeds RMS_THRESHOLD. Blocking.
void runLearningPhase() {
  Serial.println("Learning phase...");
  Serial.println("Stage 1: building reference matrix (play the target sound)");

  int refFrameIdx = 0;
  while (refFrameIdx < NUM_REF_FRAMES) {
    float samplingFreq;
    acquireAndProcessFrame(fftReal, samplingFreq);

    if (rmsValue < RMS_THRESHOLD && refFrameIdx == 0) continue;

    computeFrameMfcc(samplingFreq);
    for (int r = 0; r < NUM_KEPT_MFCCS; r++) refMatrix[r][refFrameIdx] = mfcc[r + 1];

    refFrameIdx++;
    Serial.print(".");
  }

  Serial.println("\nStage 2: estimating distance threshold (keep playing the target sound)");
  learning2Active = true;
}

void sortAscending(float *v, int n) {
  for (int i = 1; i < n; i++) {
    float key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) {
      v[j + 1] = v[j];
      j--;
    }
    v[j + 1] = key;
  }
}

// Picks the threshold as a percentile of the learning distances, lowering the
// percentile while it is too far above the smallest distances.
float computeAdaptiveThreshold(float *distances, int n) {
  sortAscending(distances, n);

  int refIdx = n > 3 ? 3 : n - 1;
  float percentile = INITIAL_PERCENTILE;
  int percentileIdx = (int)((n - 1) * percentile);
  while (distances[percentileIdx] > THRESHOLD_RATIO * distances[refIdx]) {
    percentile /= PERCENTILE_DECAY;
    percentileIdx = (int)((n - 1) * percentile);
  }
  return distances[percentileIdx];
}

// Stage 2: matches new frames against the reference matrix and collects the
// minimum distances, then derives the detection threshold. Called from loop().
void runLearning2Phase() {
  float samplingFreq;
  acquireAndProcessFrame(fftReal, samplingFreq);

  bool newSequence;
  if (!passesRmsGate(&newSequence)) return;
  if (newSequence) rebuildCount++;

  computeFrameMfcc(samplingFreq);
  updateTestMatrix();

  if (testFrameIdx < NUM_TEST_FRAMES - 1) {
    testFrameIdx++;
    Serial.print(".");
    return;
  }

  float bestDist = findBestDistance();
  testFrameIdx++;
  Serial.print(".");

  if (learning2Count < NUM_LEARNING_DISTANCES) learningDistances[learning2Count++] = bestDist;

  if (learning2Count < NUM_LEARNING_DISTANCES && rebuildCount < MAX_REBUILDS) return;

  // ---- Stage 2 complete ----
  learning2Active = false;
  distanceTh = computeAdaptiveThreshold(learningDistances, learning2Count);

  Serial.println("\nSorted learning distances:");
  for (int i = 0; i < learning2Count; i++) {
    Serial.print(i);
    Serial.print(") ");
    Serial.println(learningDistances[i], 3);
  }
  Serial.print("Distance threshold: ");
  Serial.println(distanceTh, 4);
  Serial.println("Learning complete. Monitoring...");

  delay(1000);
  learning2Count = 0;
  testFrameIdx = 0;
  isMonitoringActive = true;
  monitoringStartTime = millis();
}

// ============================================================================
// EXPERIMENT LOG (measurement protocol)
// ============================================================================

#if ENABLE_EXPERIMENT_LOG
// Prints a summary at the STOP marks and resets the statistics at the START
// marks, giving ~30 s measurement windows (times in ms since monitoring start).
const long STOP_MARKS[] = { 10000, 50000, 90000 };
const long START_MARKS[] = { 19000, 59000 };
const long MARK_WINDOW = 1000;

bool nearMark(long elapsed, const long *marks, int count) {
  for (int i = 0; i < count; i++)
    if (elapsed >= marks[i] && elapsed <= marks[i] + MARK_WINDOW) return true;
  return false;
}

void updateExperimentLog(long elapsed) {
  if (stopPending && nearMark(elapsed, STOP_MARKS, 3)) {
    Serial.print("------------ STOP at ");
    Serial.print(elapsed);
    Serial.println(" ms ------------");
    Serial.print("Distance threshold: ");
    Serial.println(distanceTh);
    Serial.print("Detection count: ");
    Serial.println(countDet);
    Serial.print("First detection time: ");
    Serial.println(firstDetTime);
    Serial.print("Second detection time: ");
    Serial.println(secondDetTime);
    Serial.print("Distinguishability index: ");
    Serial.println(minDist / distanceTh);
    Serial.println("----------------------------------------");
    stopPending = false;
  }

  if (!stopPending && nearMark(elapsed, START_MARKS, 2)) {
    Serial.print("------------ START at ");
    Serial.print(elapsed);
    Serial.println(" ms ------------");
    resetStatistics();
  }
}
#endif

void resetStatistics() {
  countDet = 0;
  firstDetTime = 0;
  secondDetTime = 0;
  minDist = 1e3;
  stopPending = true;
  isFirstDet = true;
  isSecondDet = false;
}

// ============================================================================
// MONITORING
// ============================================================================

void runMonitoringPhase() {
  long now = millis();
  long elapsed = now - monitoringStartTime;

#if ENABLE_EXPERIMENT_LOG
  updateExperimentLog(elapsed);
#endif

  float samplingFreq;
  acquireAndProcessFrame(fftReal, samplingFreq);

  bool newSequence;
  if (!passesRmsGate(&newSequence)) return;

  computeFrameMfcc(samplingFreq);
  updateTestMatrix();

  // Wait until the test matrix is full
  if (testFrameIdx < NUM_TEST_FRAMES - 1) {
    testFrameIdx++;
    return;
  }

  float bestDist = findBestDistance();
  if (bestDist < minDist) minDist = bestDist;

  if (bestDist < distanceTh) {
    lastDetectionTime = now;
    countDet++;
    testFrameIdx = -1;  // reset the test matrix after a detection

    if (isFirstDet) {
      firstDetTime = elapsed;
      isFirstDet = false;
      isSecondDet = true;
    } else if (isSecondDet) {
      secondDetTime = elapsed;
      isSecondDet = false;
    }

    Serial.print("SOUND DETECTED: distance ");
    Serial.print(bestDist);
    Serial.print(" at ");
    Serial.print(elapsed);
    Serial.println(" ms");
  }
#if LOG_DISTANCES
  else {
    Serial.print("distance ");
    Serial.print(bestDist);
    Serial.print(" at ");
    Serial.print(elapsed);
    Serial.println(" ms");
  }
#endif

  testFrameIdx++;
}

// ============================================================================
// SETUP / LOOP
// ============================================================================

void startLearning() {
  isMonitoringActive = false;
  learning2Active = false;
  learning2Count = 0;
  rebuildCount = -1;
  testFrameIdx = 0;
  monitoringStartTime = 0;
  lastDetectionTime = -1e9;
  resetStatistics();
  runLearningPhase();
}

void setup() {
  delay(1000);  // give the serial monitor time to connect
  Serial.begin(115200);
  pinMode(MIC_PIN, INPUT);
  Serial.println("=== ESP32 Sound Recognition ===");
  Serial.println("Send 'L' to start the learning phase.");
}

void loop() {
  if ((millis() - lastDetectionTime) < PAUSE_TIME_MS) return;

  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'L' || cmd == 'l') startLearning();
  }

  if (learning2Active) {
    runLearning2Phase();
  } else if (isMonitoringActive) {
    runMonitoringPhase();
  }
}
