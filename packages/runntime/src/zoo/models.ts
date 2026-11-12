/** Ready-made model configs: the Hugging Face URLs of the weights and the
 *  tokenizer, plus the options each variant needs. Pass one to a
 *  create<Task>() call, or spread it and change a field.
 *
 *  Every family has a DEFAULT, the smallest download that runs everywhere,
 *  and one entry per hosted precision or size. The files live in the
 *  software-mansion `runntime-*` repos, one per model, laid out as
 *  `<size>/<precision>/model.safetensors` with `tokenizer.json` at the
 *  root where the model has one. */

import type { SpeechToTextModel } from './tasks/audio/speechToText.ts';
import type { ImageClassifierModel } from './tasks/cv/imageClassification.ts';
import type { DepthEstimatorModel } from './tasks/cv/depthEstimation.ts';
import type { InstanceSegmenterModel } from './tasks/cv/instanceSegmentation.ts';
import type { ObjectDetectorModel } from './tasks/cv/objectDetection.ts';
import type { KeypointDetectorModel } from './tasks/cv/keypointDetection.ts';
import type { EmbedderModel } from './tasks/multimodal/embedding.ts';
import type { PrivacyFilterModel } from './tasks/nlp/privacyFilter.ts';
import type { D1DeciderModel } from './tasks/nlp/d1.ts';
import type { Lfm3ChatModel } from './tasks/nlp/lfm3.ts';
import { LFM3_Q4_K_M_SOURCE } from './lfm3/gguf.ts';

const HUB = 'https://huggingface.co/software-mansion';

const weights = (repo: string, size: string, precision: string) =>
  `${HUB}/runntime-${repo}/resolve/main/${size}/${precision}/model.safetensors`;
const tokenizer = (repo: string) => `${HUB}/runntime-${repo}/resolve/main/tokenizer.json`;

/** DEFAULT, F16 and F32 entries of one size of a model that ships both
 *  precisions. DEFAULT is the f16 file with no dtype set: half the download,
 *  and the task runs it in f16 where the device has shader-f16 and in f32
 *  elsewhere. F16 forces f16 math, F32 is the f32 file in f32. */
const precisions = <T extends object>(repo: string, size: string, common: T) => {
  const f16 = { ...common, modelPath: weights(repo, size, 'f16') };
  return {
    DEFAULT: f16,
    F16: { ...f16, dtype: 'f16' as const },
    F32: { ...common, modelPath: weights(repo, size, 'f32'), dtype: 'f32' as const },
  };
};

const MINILM = precisions('all-MiniLM-L6-v2', 'base', {
  tokenizerPath: tokenizer('all-MiniLM-L6-v2'),
} satisfies EmbedderModel);

/** EmbeddingGemma 2 ships the text model and the vision tower as two files
 *  of each precision. */
const embeddingGemma2 = (precision: string) => ({
  modelPath: weights('embeddinggemma-2', 'text', precision),
  visionPath: weights('embeddinggemma-2', 'vision', precision),
  audioPath: weights('embeddinggemma-2', 'audio', precision),
  tokenizerPath: tokenizer('embeddinggemma-2'),
});
const EMBEDDINGGEMMA_2 = {
  DEFAULT: embeddingGemma2('f16'),
  F16: { ...embeddingGemma2('f16'), dtype: 'f16' as const },
  F32: { ...embeddingGemma2('f32'), dtype: 'f32' as const },
} satisfies Record<string, EmbedderModel>;

const MOONSHINE_TINY = precisions('moonshine', 'tiny', {
  arch: 'moonshine',
  tokenizerPath: tokenizer('moonshine'),
} satisfies SpeechToTextModel);
const MOONSHINE_BASE = precisions('moonshine', 'base', {
  arch: 'moonshine',
  tokenizerPath: tokenizer('moonshine'),
} satisfies SpeechToTextModel);
const MOONSHINE_STREAMING_TINY = precisions('moonshine-streaming', 'tiny', {
  arch: 'moonshine-streaming',
  tokenizerPath: tokenizer('moonshine-streaming'),
} satisfies SpeechToTextModel);

const PRIVACY_FILTER_INT8 = {
  modelPath: weights('privacy-filter', 'base', 'int8g640'),
} satisfies PrivacyFilterModel;

const YOLO26_N = { modelPath: weights('yolo26', 'n', 'f16') } satisfies ObjectDetectorModel;
const YOLO26_S = { modelPath: weights('yolo26', 's', 'f16') } satisfies ObjectDetectorModel;
const YOLO26_M = { modelPath: weights('yolo26', 'm', 'f16') } satisfies ObjectDetectorModel;

const MOBILENETV4_SMALL = {
  modelPath: weights('mobilenetv4', 'small', 'f16'),
} satisfies ImageClassifierModel;

const YOLO26_SEG_N = {
  modelPath: weights('yolo26-seg', 'n', 'f16'),
} satisfies InstanceSegmenterModel;
const YOLO26_SEG_S = {
  modelPath: weights('yolo26-seg', 's', 'f16'),
} satisfies InstanceSegmenterModel;
const YOLO26_SEG_M = {
  modelPath: weights('yolo26-seg', 'm', 'f16'),
} satisfies InstanceSegmenterModel;

const YOLO26_POSE_N = {
  modelPath: weights('yolo26-pose', 'n', 'f16'),
} satisfies KeypointDetectorModel;
const YOLO26_POSE_S = {
  modelPath: weights('yolo26-pose', 's', 'f16'),
} satisfies KeypointDetectorModel;
const YOLO26_POSE_M = {
  modelPath: weights('yolo26-pose', 'm', 'f16'),
} satisfies KeypointDetectorModel;

const DEPTHART_S = { modelPath: weights('depthart', 's', 'f16') } satisfies DepthEstimatorModel;
const DEPTHART_B = { modelPath: weights('depthart', 'b', 'f16') } satisfies DepthEstimatorModel;

const D1_REVISION = '02b55d7076f15129e59ab3f94783f32c4b088674';
const D1_TOKENIZER = `https://huggingface.co/LiquidAI/d1-omni-600M/resolve/${D1_REVISION}/tokenizer.json`;
const D1_TEMPERATURES = {
  'choice:11+': 1.372515082359314,
  'choice:2': 1.7465145587921143,
  'choice:3-5': 1.3998981714248657,
  'choice:6-10': 1.1751071214675903,
  'noul:2': 1.6663223505020142,
  'score:3-5': 1.7301132678985596,
  'score:6-10': 1.0,
  choice: 1.0,
  score: 1.0,
  noul: 1.0,
};
const D1_Q8 = {
  tokenizerPath: D1_TOKENIZER,
  temperatures: D1_TEMPERATURES,
} satisfies D1DeciderModel;
const D1_F32 = {
  ...D1_Q8,
  modelPath: `https://huggingface.co/LiquidAI/d1-omni-600M/resolve/${D1_REVISION}/model.safetensors`,
} satisfies D1DeciderModel;

const LFM3_REVISION = '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67';
const LFM3_OMNI_2B = {
  modelPath: `https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/${LFM3_REVISION}/model.safetensors`,
  tokenizerPath: `https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/${LFM3_REVISION}/tokenizer.json`,
} satisfies Lfm3ChatModel;
const LFM3_OMNI_2B_Q4_K_M = {
  format: 'gguf',
  modelPath: LFM3_Q4_K_M_SOURCE.url,
  tokenizerPath: LFM3_OMNI_2B.tokenizerPath,
} satisfies Lfm3ChatModel;

export const models = {
  lfm3: {
    /** Liquid LFM3-Omni 2B text backbones. DEFAULT remains the original BF16 weights. */
    LFM3_OMNI_2B: {
      DEFAULT: LFM3_OMNI_2B,
      /** Published Liquid Q4_K_M GGUF; uses the pinned tokenizer without loading floating weights. */
      Q4_K_M: LFM3_OMNI_2B_Q4_K_M,
    },
  },
  d1: {
    /** Liquid's D1 Omni 600M text decision model. Q8 weights are generated locally. */
    D1_OMNI_600M: {
      /** Original float checkpoint. */
      DEFAULT: D1_F32,
      F32: D1_F32,
      /** Supply a locally converted native Q8 checkpoint as modelPath. */
      Q8: D1_Q8,
    },
  },
  embedding: {
    /** Google's EmbeddingGemma 2: text in 100+ languages and code, images
     *  and video in one 768-dim space, inputs up to 2048 tokens. 270M
     *  parameters in the text model, 170M in the vision tower. */
    EMBEDDINGGEMMA_2: EMBEDDINGGEMMA_2,
    /** all-MiniLM-L6-v2: English text in 384-dim vectors, 22M parameters. */
    ALL_MINILM_L6_V2: MINILM,
  },
  privacyFilter: {
    /** OpenAI's privacy-filter: 1.5B parameters, 50M active per token,
     *  int8 weights in groups of 640. The one hosted format. */
    PRIVACY_FILTER: { DEFAULT: PRIVACY_FILTER_INT8 },
  },
  speechToText: {
    /** English. */
    MOONSHINE: {
      /** 27M parameters. */
      TINY: MOONSHINE_TINY,
      /** 61M parameters. More accurate, slower. */
      BASE: MOONSHINE_BASE,
    },
    /** English, the streaming family. */
    MOONSHINE_STREAMING: {
      /** 34M parameters. */
      TINY: MOONSHINE_STREAMING_TINY,
    },
  },
  objectDetection: {
    /** YOLO26 detect, COCO-80 classes, f16 weights. */
    YOLO26: {
      DEFAULT: YOLO26_N,
      /** The nano size, 5 MB. */
      N: YOLO26_N,
      /** The small size, 18 MB. */
      S: YOLO26_S,
      /** The medium size, 39 MB. More accurate, slower. */
      M: YOLO26_M,
    },
  },
  instanceSegmentation: {
    /** YOLO26 segment, COCO-80 classes with a mask per object, f16 weights. */
    YOLO26_SEG: {
      DEFAULT: YOLO26_SEG_N,
      /** The nano size, 5 MB. */
      N: YOLO26_SEG_N,
      /** The small size, 20 MB. */
      S: YOLO26_SEG_S,
      /** The medium size, 45 MB. More accurate, slower. */
      M: YOLO26_SEG_M,
    },
  },
  keypointDetection: {
    /** YOLO26 pose, people with 17 COCO landmarks, f16 weights. */
    YOLO26_POSE: {
      DEFAULT: YOLO26_POSE_N,
      /** The nano size, 6 MB. */
      N: YOLO26_POSE_N,
      /** The small size, 20 MB. */
      S: YOLO26_POSE_S,
      /** The medium size, 42 MB. More accurate, slower. */
      M: YOLO26_POSE_M,
    },
  },
  depthEstimation: {
    /** DepthART relative depth at 448×448, f16 weights. */
    DEPTHART: {
      DEFAULT: DEPTHART_S,
      /** The small size, 13 MB. */
      S: DEPTHART_S,
      /** The base size, 24 MB. More detail, slower. */
      B: DEPTHART_B,
    },
  },
  imageClassification: {
    /** MobileNetV4 conv, ImageNet-1k, f16 weights. The model runs in f16
     *  only, so each size has one entry. */
    MOBILENETV4: {
      DEFAULT: MOBILENETV4_SMALL,
      /** The small size, 3.8M parameters, 10 MB. */
      SMALL: MOBILENETV4_SMALL,
    },
  },
} as const;
