import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { getCosineSimilarity } from "./functions";

export interface ImagePrediction {
  className: string;
  probability: number;
}

export interface ImageScore {
  url: string;
  score: number;
}

export class ImageProcessor {
  constructor(
    private model: mobilenet.MobileNet | null = null,
    private userPreferenceVector: tf.Tensor | null = null,
    private currentImageEmbedding: tf.Tensor | null = null,
    private metadata: Record<string, { label: string }> = {},
    private numRatings: number = 0
  ) {}

  async processImage(imageData: tf.Tensor3D) {
    if (!this.model) {
      throw new Error("Model not initialized");
    }

    if (imageData.shape[0] === 0 || imageData.shape[1] === 0) {
      throw new Error("Invalid image dimensions");
    }

    const predictions = await this.model.classify(imageData);
    const embeddingModel = this.model as any;
    this.currentImageEmbedding = embeddingModel.infer(imageData, true);

    return {
      predictions,
      embedding: this.currentImageEmbedding
    };
  }

  getKeywordScore(labelA: string, labelB: string): number { // Calculate keyword score based on shared words
    if (!labelA || !labelB) return 0;
    const aWords = labelA.split(/[ ,]+/);
    const bWords = labelB.split(/[ ,]+/);
    const shared = aWords.filter(word => bWords.includes(word));
    return shared.length > 0 ? 1 : 0;
  }

  calculateImageScore(embedding: tf.Tensor, uploadedLabel: string, targetLabel: string): number {
    let score = 0;

    if (this.userPreferenceVector && this.currentImageEmbedding) {

      // Use both user preference and current image embedding
      // Hybrid scoring: combine user preferences and content-similarity, this is 70%user preference and 30% current image, user preference matters more
      const similarityWithPreference = getCosineSimilarity(embedding, this.userPreferenceVector);
      const similarityWithCurrentImage = getCosineSimilarity(embedding, this.currentImageEmbedding);

      // Weighted hybrid score (tweak weights as needed)
      score = 0.7 * similarityWithPreference + 0.3 * similarityWithCurrentImage;

    } else if (this.userPreferenceVector) {

      // Use only user preference
      score = getCosineSimilarity(embedding, this.userPreferenceVector);

    } else if (this.currentImageEmbedding) {

      // Use only current image embedding, mainly used at start when there is no user preference determined by rating
      score = getCosineSimilarity(embedding, this.currentImageEmbedding);

    }else {

      // Fallback (shouldnt hit)
      score = 0;
      
    }

    const keywordScore = this.getKeywordScore(uploadedLabel, targetLabel);
    const contextualBoost = uploadedLabel && targetLabel.includes(uploadedLabel) ? 0.1 : 0;
    return 0.8 * score + 0.2 * keywordScore + contextualBoost;
  }

  getSimilarityThreshold(): number {
    return Math.min(0.1 + 0.05 * this.numRatings, 0.6);
  }
}