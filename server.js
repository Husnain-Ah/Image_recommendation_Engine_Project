const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const fuzz = require('fuzzball');

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

app.post('/search-images', (req, res) => {
  const { keyword } = req.body;
  if (!keyword) return res.status(400).json({ error: 'No keyword provided' });

  const uniqueLabelsInIndex = [...new Set(imageIndex.map(img => img.label))];

  const bestMatch = fuzz.extract(keyword, uniqueLabelsInIndex, {
    scorer: fuzz.token_set_ratio,
    returnObjects: true,
  });

  console.log('Fuzzy search results:', bestMatch);

  const topLabel = bestMatch[0].choice;
  console.log(`Searching for images similar to: "${keyword}" → "${topLabel}"`);

  const results = imageIndex
    .filter(img => img.label === topLabel)
    .slice(0, 10)
    .map(img => `tiny-imagenet-200${img.path}`);

  if (results.length === 0) {
    return res.status(404).json({ error: `No images found for label "${topLabel}"` });
  }

  res.json({ results });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  loadLabelMap();
  loadImages();
});
