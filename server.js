const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const fuzz = require('fuzzball');
const axios = require('axios'); 

async function getSemanticEmbedding(text) {
  try {
    const response = await axios.post('http://localhost:5001/embed', { text });
    return response.data.embedding;
  } catch (err) {
    console.error('Embedding error:', err.message);
    return null;
  }
}

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.use('/tiny-imagenet-200', express.static(path.join(__dirname, 'tiny-imagenet-200')));

app.get('/test', (req, res) => {
  res.status(200).send('Server is up and running!');
});

const DATASET_PATH = path.join(__dirname, 'tiny-imagenet-200');
const TRAIN_IMAGES_PATH = path.join(DATASET_PATH, 'train');
const VAL_IMAGES_PATH = path.join(DATASET_PATH, 'val', 'images');

let imageIndex = []; 
let labelMap = {};   

function loadLabelMap() {
  const filePath = path.join(DATASET_PATH, 'words.txt');
  const lines = fs.readFileSync(filePath, 'utf-8').trim().split('\n');
  for (const line of lines) {
    const [wnid, label] = line.split('\t');
    labelMap[wnid] = label;
  }
  console.log(`Loaded ${Object.keys(labelMap).length} labels`);
}

function loadImages() {
  const wnids = fs.readFileSync(path.join(DATASET_PATH, 'wnids.txt'), 'utf-8').trim().split('\n');
  let total = 0;

  for (const wnid of wnids) {
    const dir = path.join(TRAIN_IMAGES_PATH, wnid, 'images');
    if (!fs.existsSync(dir)) continue;
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.JPEG'));
    for (const file of files) {
      imageIndex.push({
        path: `/train/${wnid}/images/${file}`,
        label: labelMap[wnid] || wnid,
      });
    }
    total += files.length;
  }

  const valAnnotations = fs.readFileSync(path.join(DATASET_PATH, 'val', 'val_annotations.txt'), 'utf-8').trim().split('\n');
  for (const line of valAnnotations) {
    const [filename, wnid] = line.split('\t');
    imageIndex.push({
      path: `/val/images/${filename}`,
      label: labelMap[wnid] || wnid,
    });
  }
  total += valAnnotations.length;

  console.log(`Total images indexed: ${total}`);
}

const cosineSimilarity = (vecA, vecB) => {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
};

app.post('/search-images', async (req, res) => {
  const { keyword } = req.body;
  if (!keyword) return res.status(400).json({ error: 'No keyword provided' });

  console.log(`Search query received: "${keyword}"`);

  const userEmbedding = await getSemanticEmbedding(keyword);
  if (!userEmbedding) return res.status(500).json({ error: 'Failed to get embedding' });

  const uniqueLabels = [...new Set(imageIndex.map(img => img.label.toLowerCase()))];

  let bestMatch = null;
  let highestSim = -1;

  for (const label of uniqueLabels) {
    const labelEmbedding = await getSemanticEmbedding(label);
    const similarity = cosineSimilarity(userEmbedding, labelEmbedding);

    if (similarity > highestSim) {
      highestSim = similarity;
      bestMatch = label;
    }
  }

  if (!bestMatch) {
    console.log(`No match found for "${keyword}"`);
    return res.status(404).json({ error: `No match found for "${keyword}"` });
  }

  console.log(`Best match for "${keyword}": "${bestMatch}" with similarity: ${highestSim.toFixed(3)}`);

  const results = imageIndex
    .filter(img => img.label.toLowerCase() === bestMatch)
    .slice(0, 10)
    .map(img => `tiny-imagenet-200${img.path}`);

  console.log(`Found ${results.length} result(s) for "${keyword}"`);

  res.json({ results, match: bestMatch, similarity: highestSim.toFixed(3) });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  loadLabelMap();
  loadImages();
});
