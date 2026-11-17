/** runntime/zoo root: what a zoo user calls. Setup, load options, errors
 *  and the input and result types every task shares. Each task runner
 *  (create<Task>) adds its export here. Building blocks under tasks/ stay
 *  internal; the package imports them by path. */

export {
  bufferSource,
  createOpfsCache,
  initRunntime,
  isRunntimeError,
  RUNNTIME_ERROR_CODES,
  RunntimeError,
  type RangeSource,
  type RunntimeErrorCode,
  type WeightCache,
} from '../core/index.ts';

export type { LoadOptions, ModelPath } from './load.ts';
export type {
  EncodeOptions,
  GpuEncoder,
  GpuImage,
  GpuImageTarget,
  GpuTensor,
} from './tasks/gpu.ts';
export { models } from './models.ts';

export {
  createEmbedder,
  similarity,
  type EmbedInput,
  type EmbedOptions,
  type EmbedPart,
  type Embedder,
  type EmbedderModel,
  type EmbeddingTask,
  type ImageTokenBudget,
} from './tasks/multimodal/embedding.ts';

export {
  createPrivacyFilter,
  type PrivacyFilter,
  type PrivacyFilterModel,
  type PrivacySpan,
} from './tasks/nlp/privacyFilter.ts';
export { createD1Decider, type D1Decider, type D1DeciderModel } from './tasks/nlp/d1.ts';
export type { Answer, Question, QuestionType } from './d1/prompt.ts';
export {
  createLfm3Chat,
  type Lfm3ChatModel,
  type Lfm3ChatOptions,
  type Lfm3ChatSession,
  type Lfm3Generation,
  type Lfm3GenerationResult,
  type Lfm3Message,
} from './tasks/nlp/lfm3.ts';
export {
  createLfm3Audio,
  createLfm3Audio as createLfm3Omni,
  type Lfm3Audio,
  type Lfm3Audio as Lfm3Omni,
  type Lfm3AudioModel,
  type Lfm3SpeechGeneration,
  type Lfm3SpeechOptions,
  type Lfm3SpeechResult,
  type Lfm3SpeechVoice,
  type Lfm3TranscriptionOptions,
} from './tasks/audio/lfm3.ts';

export {
  createSpeechToText,
  type SpeechStreamUpdate,
  type SpeechToText,
  type SpeechToTextArch,
  type SpeechToTextModel,
} from './tasks/audio/speechToText.ts';
export { decodeAudio, resampleAudio, SPEECH_SAMPLE_RATE } from './tasks/audio/audioInput.ts';

export {
  createObjectDetector,
  type DetectObjectsOptions,
  type EncodeDetectObjectsOptions,
  type ObjectDetection,
  type ObjectDetectionData,
  type ObjectDetectionList,
  type ObjectDetector,
  type ObjectDetectorModel,
} from './tasks/cv/objectDetection.ts';
export type { Yolo26Variant } from './yolo26/config.ts';
export type { Yolo26Input } from './yolo26/detector.ts';

export {
  createInstanceSegmenter,
  type EncodeSegmentInstancesOptions,
  GPU_MASK_SIZE,
  type InstanceSegmentation,
  type InstanceSegmentationData,
  type InstanceSegmentationList,
  type InstanceSegmenter,
  type InstanceSegmenterModel,
  type SegmentInstancesOptions,
} from './tasks/cv/instanceSegmentation.ts';

export {
  createKeypointDetector,
  type DetectKeypointsOptions,
  type EncodeDetectKeypointsOptions,
  type KeypointDetection,
  type KeypointDetectionData,
  type KeypointDetectionList,
  type KeypointDetector,
  type KeypointDetectorModel,
  type Landmark,
} from './tasks/cv/keypointDetection.ts';
export { COCO_LANDMARKS, type CocoLandmark } from './yolo26/pipeline.ts';

export {
  createDepthEstimator,
  type DepthEstimator,
  type DepthEstimatorModel,
  type DepthInput,
  type DepthMap,
  type DepthMapData,
  type EncodeEstimateDepthOptions,
} from './tasks/cv/depthEstimation.ts';
export type { DepthartVariant } from './depthart/estimator.ts';

export {
  createImageClassifier,
  type ClassificationList,
  type ClassifyInput,
  type ClassifyOptions,
  type EncodeClassifyOptions,
  type ImageClassifier,
  type ImageClassifierModel,
} from './tasks/cv/imageClassification.ts';
export { type Classification } from './mobilenetv4/pipeline.ts';

export {
  imageBuffer,
  imageBufferFromImageData,
  type ImageBuffer,
  type ImageFormat,
  type ResizeMode,
} from './tasks/cv/image.ts';
export type { Point, Size } from './tasks/cv/ops/point.ts';
export type { BoundingBox, BoxFormat, BoxMap } from './tasks/cv/ops/box.ts';
