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
let metadata = {};
let invertedIndex = {};




function loadLabelMap() {
  const filePath = path.join(DATASET_PATH, 'words.txt');
  const lines = fs.readFileSync(filePath, 'utf-8').trim().split('\n');
  for (const line of lines) {
    const [wnid, label] = line.split('\t');
    labelMap[wnid] = label;
  }
  console.log(`Loaded ${Object.keys(labelMap).length} labels`);
}


function loadMetadataAndBuildInvertedIndex() {
  const metadataPath = path.join(__dirname, 'annoy_data', 'metadata.json');  // Update path as needed
  if (!fs.existsSync(metadataPath)) {
    console.error('metadata.json not found.');
    return;
  }

  metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));

  // Build inverted index
  for (const item of metadata) {
    const label = item.label.toLowerCase();
    if (!invertedIndex[label]) {
      invertedIndex[label] = [];
    }
    invertedIndex[label].push(item.filename);
  }

  console.log('Inverted index built with', Object.keys(invertedIndex).length, 'labels.');
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

  const userEmbedding = await getSemanticEmbedding(keyword);
  if (!userEmbedding) return res.status(500).json({ error: 'Failed to get embedding' });

  const uniqueLabels = Object.keys(invertedIndex);  

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
    return res.status(404).json({ error: `No match found for "${keyword}"` });
  }

  const matchingImages = invertedIndex[bestMatch] || [];
  console.log(`Query: "${keyword}" matched label "${bestMatch}"`);
  console.log(`Found ${matchingImages.length} matching images.`);

  const results = matchingImages.slice(0, 10).map(filename => {
    const imgMeta = metadata.find(meta => meta.filename === filename);
    return `tiny-imagenet-200/train/${imgMeta.wnid}/images/${filename}`;
  });

  res.json({ results, match: bestMatch, similarity: highestSim.toFixed(3) });
});



app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  loadLabelMap();
  loadImages();
  loadMetadataAndBuildInvertedIndex();
});
