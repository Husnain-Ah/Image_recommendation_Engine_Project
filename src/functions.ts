import * as tf from '@tensorflow/tfjs';

export function getCosineSimilarity(tensorA: tf.Tensor, tensorB: tf.Tensor): number {
  const dotProduct = tf.sum(tf.mul(tensorA, tensorB));
  const normA = tf.norm(tensorA);
  const normB = tf.norm(tensorB);
  const similarity = dotProduct.div(normA.mul(normB));
  return similarity.dataSync()[0];
}