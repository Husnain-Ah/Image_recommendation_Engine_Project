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

    processor = new ImageProcessor(mockModel, null, null, {}, 0);
  });

  describe('processImage tests', () => {
    test('should throw if model is not initialised yet', async () => {
      const Processor = new ImageProcessor(null, null, null, {}, 0);
      const Tensor = tf.tensor3d([0], [1, 1, 1]);

      await expect(Processor.processImage(Tensor)).rejects.toThrow('Model not initialized');
    });

    test('should fail on empty tensor shape', async () => {
      const tensor = tf.tensor3d([], [0, 0, 3]);
      await expect(processor.processImage(tensor)).rejects.toThrow('Invalid image dimensions');
    });

    test('should return predictions and embedding if valid', async () => {
      const tensor = tf.tensor3d([1, 2, 3], [1, 1, 3]);
      const mockPredictions = [{ className: 'car', probability: 0.9 }];
      const mockEmbedding = tf.tensor1d([0.1, 0.2, 0.3]);

      mockModel.classify.mockResolvedValue(mockPredictions);
      mockModel.infer.mockReturnValue(mockEmbedding);

      const result = await processor.processImage(tensor);

      expect(result).toHaveProperty('predictions');
      expect(result.predictions).toEqual(mockPredictions);

      expect(result.embedding).toBeDefined();
    });
  });

  describe('getKeywordScore tests', () => {
    test('should return 0 if either label is missing', () => {
      expect(processor.getKeywordScore('', 'dog')).toBe(0);
      expect(processor.getKeywordScore('cat', '')).toBe(0);
    });

    test('should return 1 when labels share common terms', () => {
      expect(processor.getKeywordScore('dog', 'dog running')).toBe(1);
    });

    test('should return 0 for unrelated terms', () => {
      expect(processor.getKeywordScore('floor', 'dragon')).toBe(0);
    });
  });

  describe('calculateImageScore tests', () => {
    test('calculates the score with user preference', () => {
      const embedding = tf.tensor1d([1, 1, 1]);
      const userPreferences = tf.tensor1d([1, 2, 3]);
      const current = tf.tensor1d([2, 3, 4]);

      processor = new ImageProcessor(mockModel, userPreferences, current, {}, 0);

      const score = processor.calculateImageScore(embedding, 'cat', 'cat playing');
      expect(typeof score).toBe('number');
      expect(score).toBeGreaterThanOrEqual(0);
    });

    test('contextualboost boosts score when labels are matched', () => {
      const embedding = tf.tensor1d([1, 2, 3]);

      const s1 = processor.calculateImageScore(embedding, 'bird', 'car driving');
      const s2 = processor.calculateImageScore(embedding, 'bird', 'bird food');

      expect(s2).toBeGreaterThan(s1);
    });
  });

  describe('getSimilarityThreshold tests', () => {
    test('uses base threshold when no ratings given (0.1)', () => {
      expect(processor.getSimilarityThreshold()).toBeCloseTo(0.1);
    });

    test('grows threshold with rating count', () => {
      processor = new ImageProcessor(mockModel, null, null, {}, 5);
      expect(processor.getSimilarityThreshold()).toBeGreaterThan(0.1);
    });

    test('caps threshold after a point (0.6)', () => {
      processor = new ImageProcessor(mockModel, null, null, {}, 100);
      expect(processor.getSimilarityThreshold()).toBeLessThanOrEqual(0.6);
    });
  });
});
