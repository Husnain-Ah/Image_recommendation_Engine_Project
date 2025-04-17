import { ImageProcessor } from '../ImageProcessor';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

describe('ImageProcessor', () => {
  let processor: ImageProcessor;
  let mockModel: jest.Mocked<mobilenet.MobileNet>;

  beforeEach(() => {
    jest.clearAllMocks();

    mockModel = {
      classify: jest.fn(),
      infer: jest.fn(),
    } as any;

    processor = new ImageProcessor(
      mockModel,
      null,
      null,
      {},
      0
    );
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('processImage tests', () => {
    test('throws error when model is not initialized', async () => {
      processor = new ImageProcessor(null, null, null, {}, 0);
      const tensor = tf.tensor3d([1, 1, 1], [1, 1, 3]);
      await expect(processor.processImage(tensor)).rejects.toThrow('Model not initialized');
    });

    test('throws error for invalid image dimensions', async () => {
      const tensor = tf.tensor3d([], [0, 0, 3]);
      await expect(processor.processImage(tensor)).rejects.toThrow('Invalid image dimensions');
    });

    test('processes valid image successfully', async () => {
      const tensor = tf.tensor3d([1, 1, 1], [1, 1, 3]);
      const mockPredictions = [{ className: 'dog', probability: 0.8 }];
      const mockEmbedding = tf.tensor1d([1, 2, 3]);

      mockModel.classify.mockResolvedValue(mockPredictions);
      mockModel.infer.mockReturnValue(mockEmbedding);

      const result = await processor.processImage(tensor);
      expect(result.predictions).toEqual(mockPredictions);
      expect(result.embedding).toBeDefined();
    });
  });

  describe('getKeywordScore tests', () => {
    test('returns 0 when either label is empty', () => {
      expect(processor.getKeywordScore('', 'dog')).toBe(0);
      expect(processor.getKeywordScore('cat', '')).toBe(0);
      expect(processor.getKeywordScore('', '')).toBe(0);
    });

    test('returns 1 when labels share words', () => {
      expect(processor.getKeywordScore('black dog', 'dog running')).toBe(1);
      expect(processor.getKeywordScore('cat playing', 'playing ball')).toBe(1);
    });

    test('returns 0 when labels have no shared words', () => {
      expect(processor.getKeywordScore('black cat', 'white dog')).toBe(0);
    });
  });

  describe('calculateImageScore tests' , () => {
    test('calculates score with user preference and current image', () => {
      const embedding = tf.tensor1d([1, 2, 3]);
      const userPref = tf.tensor1d([2, 3, 4]);
      const currentEmb = tf.tensor1d([3, 4, 5]);

      processor = new ImageProcessor(
        mockModel,
        userPref,
        currentEmb,
        {},
        0
      );

      const score = processor.calculateImageScore(embedding, 'dog', 'dog running');
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1.1);
      expect(tf.mul).toHaveBeenCalled();
    });

    test('includes contextual boost when labels match', () => {
      const embedding = tf.tensor1d([1, 2, 3]);
      const score1 = processor.calculateImageScore(embedding, 'dog', 'cat running');
      const score2 = processor.calculateImageScore(embedding, 'dog', 'dog running');
      expect(score2).toBeGreaterThan(score1);
    });
  });

  describe('getSimilarityThreshold tests', () => {
    test('returns initial threshold when no ratings', () => {
      expect(processor.getSimilarityThreshold()).toBe(0.1);
    });

    test('increases threshold with ratings', () => {
      processor = new ImageProcessor(mockModel, null, null, {}, 5);
      expect(processor.getSimilarityThreshold()).toBe(0.35);
    });

    test('caps threshold at maximum', () => {
      processor = new ImageProcessor(mockModel, null, null, {}, 20);
      expect(processor.getSimilarityThreshold()).toBe(0.6);
    });
  });
});