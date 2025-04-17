import * as tf from "@tensorflow/tfjs"
import * as mobilenet from "@tensorflow-models/mobilenet"

import { searchLocalImages } from "./imageSearch";
import { renderUserVectorChart } from "./barsAndCharts";
import { renderForceGraph } from "./barsAndCharts";
import { ImageProcessor } from './ImageProcessor';

const imageInput = document.getElementById("image-input")! as HTMLInputElement
const imageDisplay = document.getElementById("image-display")! as HTMLImageElement
const resultsDiv = document.getElementById("results")!
const loadingDiv = document.getElementById("loading")!
const imageGallery = document.getElementById("image-gallery")!

let userPreferenceVector: tf.Tensor | null = null;
let currentImageEmbedding: tf.Tensor | null = null;
let predictions: any[] = [];
let numRatings = 0;

let model: mobilenet.MobileNet | null = null
let imageProcessor: ImageProcessor;

async function initializeBackend() { // Initialize TensorFlow.js backend
  try {
    await tf.setBackend("webgl")
    await tf.ready()
    console.log("TensorFlow.js backend initialized:", tf.getBackend())
  } catch (error) {
    console.error("Error initializing TensorFlow.js backend:", error)
    resultsDiv.innerText = "Error initializing TensorFlow.js backend."
  }
}

async function loadModel() { // Load MobileNetV2 model
  console.log("Loading MobileNetV2 model...")
  loadingDiv.style.display = "block"

  try {
    model = await mobilenet.load({ version: 2, alpha: 1.0 }); //  MobileNetV2 to match annoy feature extraction
    console.log("Model loaded successfully.")
    resultsDiv.innerText = "Model loaded. Ready for image selection."
  } catch (error) {
    console.error("Error loading model:", error)
    resultsDiv.innerText = "Error loading model. Please try again later."
  } finally {
    loadingDiv.style.display = "none"
  }
}

const BASE_URL = "http://localhost:3000";
let metadata: Record<string, { label: string }> = {};

async function handleImageUpload(event: Event, imageProcessor: ImageProcessor) { // Handle image upload and processing with MobileNet
  const target = event.target as HTMLInputElement
  if (!target.files || target.files.length === 0) {
    console.log("No file selected.")
    return
  }

  const file = target.files[0]
  const reader = new FileReader()

  reader.onload = (e) => {
    if (!e.target?.result) {
      resultsDiv.innerText = "Error reading file."
      return
    }

    imageDisplay.src = e.target.result as string
    imageDisplay.style.display = "block"
    resultsDiv.innerText = "Image loaded. Processing..."
    loadingDiv.style.display = "block"
  }

  reader.onerror = () => {
    console.error("Error reading file")
    resultsDiv.innerText = "Error reading the selected file."
  }

  reader.readAsDataURL(file)

  imageDisplay.onload = async () => {
    try {
      const tensor = tf.browser.fromPixels(imageDisplay);
      const { predictions, embedding } = await imageProcessor.processImage(tensor);
      currentImageEmbedding = embedding;
      displayResults(predictions);
      
      if (predictions && predictions.length > 0) {
        const topPrediction = predictions[0].className.split(",")[0].trim();
        const imageUrls = await searchLocalImages(topPrediction);
        await displayImageResults(imageUrls);
      }
    } catch (error) {
      console.error("Error during classification:", error);
      resultsDiv.innerText = "Error processing image: " + (error as Error).message;
    } finally {
      loadingDiv.style.display = "none";
    }
  };
}

function displayResults(predictions: any) { // Display classification results 
  if (!predictions || predictions.length === 0) {
    resultsDiv.innerText = "No classification results."
    return
  }

  let resultText = "Predictions:<br><ul>";
  predictions.forEach((prediction: { className: string; probability: number }) => {
    resultText += `<li><span class="prediction-label">${prediction.className}</span><span class="prediction-value">${(prediction.probability * 100).toFixed(2)}%</span></li>`;
  });
  resultText += "</ul>";
  resultsDiv.innerHTML = resultText;
  resultsDiv.style.display = "block";
}

async function loadMetadata() {
  if (Object.keys(metadata).length === 0) {
    const res = await fetch(`${BASE_URL}/annoy_data/metadata.json`);
    const json = await res.json();
    for (const item of json) {
      metadata[item.filename] = {
        label: item.label.toLowerCase().trim()
      };
    }
  }
}

function loadPromise(img: HTMLImageElement): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${img.src}`));
  });
}


function displayTopImages(topImages: { url: string; score: number }[]) {
  for (const { url, score } of topImages) {
    const img = document.createElement("img");
    img.src = `${BASE_URL}/${url}`;
    img.alt = `Score: ${score.toFixed(2)}`;
    img.onerror = () => {
      console.error("Failed to load image:", img.src);
      img.style.display = "none";
    };
    imageGallery.appendChild(img);
  }
  imageGallery.style.display = "block";
  console.log("Final filtered image URLs:", topImages.map(i => i.url));
}

async function displayImageResults(imageUrls: string[]) { // Display related images in the gallery
  imageGallery.innerHTML = "";

  const spinner = document.getElementById("search-loading")!;
  spinner.style.display = "block"; 

  if (!imageUrls || imageUrls.length === 0) {
    imageGallery.innerHTML = "<p>No related images were found</p>";
    spinner.style.display = "none"; 
    return;
  }

  await loadMetadata();

  const imageScores: { url: string; score: number }[] = [];
  const uploadedLabel = predictions[0]?.className.toLowerCase().trim();

  for (const url of imageUrls) {
    const filename = url.split("/").pop()!;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `${BASE_URL}/${url}`;

    try {
      const loadedImg = await loadPromise(img);
      const tensor = tf.browser.fromPixels(loadedImg);
      const embeddingModel = model as any;
      const embedding = embeddingModel.infer(tensor, true) as tf.Tensor;

      const labelB = metadata[filename]?.label ?? "";
      const score = imageProcessor.calculateImageScore(embedding, uploadedLabel, labelB);
      imageScores.push({ url, score });
    } catch (err) {
      console.error(err);
    } 
  }

  spinner.style.display = "none";

  // top k filtering gives top 6 images after filtering by similarity threshold
  const k = 6; // Number of top images to display
  const MAX_PER_LABEL = 15; // Maximum images per label to show in force graph
  const labelGroups: Record<string, typeof imageScores> = {};
  
  //low original thresholds, dynamicallly increases with ratings to get better recommendations
  const SIMILARITY_THRESHOLD = imageProcessor.getSimilarityThreshold();

  // Filter images by similarity threshold first
  const filteredImageScores = imageScores
    .filter(img => img.score >= SIMILARITY_THRESHOLD)
    .sort((a, b) => b.score - a.score);

  // Group by label
  for (let img of filteredImageScores) { 
    const filename = img.url.split("/").pop()!;
    const label = metadata[filename]?.label ?? "Unknown";

    if (!labelGroups[label]) labelGroups[label] = [];
    if (labelGroups[label].length < MAX_PER_LABEL) {
      labelGroups[label].push(img);
    }
  }

  const topImages = Object.values(labelGroups).flat().slice(0, k);
  const forceImages = Object.values(labelGroups).flat().slice(0, MAX_PER_LABEL);

  displayTopImages(topImages);
  showRatingSection();
  renderForceGraph(imageDisplay.src, forceImages);
}

function showRatingSection() { // Show the rating section after image processing
  const ratingSection = document.getElementById('rating-section');
  if (ratingSection) {
    ratingSection.style.display = 'block';
  }
}

document.getElementById('submit-rating')!.addEventListener('click', async () => { // Submit rating button
  const rating = parseInt((document.getElementById('rating-input') as HTMLInputElement).value);

  if (rating >= 1 && rating <= 10) {
    console.log('User rating:', rating);
    numRatings++;

    if (currentImageEmbedding) {
      const weight = rating / 10;
      const scaledEmbedding = currentImageEmbedding.mul(tf.scalar(weight));

      if (userPreferenceVector) {
        userPreferenceVector = tf.add(userPreferenceVector, scaledEmbedding);
      } else {
        userPreferenceVector = scaledEmbedding.clone();
      }

      console.log("Updated user preference vector:", userPreferenceVector.arraySync());
    }
    if (userPreferenceVector) {
      renderUserVectorChart(userPreferenceVector);
    }
    

    alert('Thank you for your rating!');
  } else {
    alert('Please enter a rating between 1 and 10');
  }
});

document.getElementById("reset-preferences")!.addEventListener("click", () => { // Reset user preferences button
  if (userPreferenceVector) {
    userPreferenceVector.dispose();
    userPreferenceVector = null;
  }
  numRatings = 0; //recet rating counter for dynamic threshold as well
  alert("User preference vector has been reset.");
});


initializeBackend().then(async () => {
  await loadModel();
  console.log("Model loaded and ready");
  
  imageProcessor = new ImageProcessor(model, userPreferenceVector, currentImageEmbedding, metadata, numRatings);
  
  imageInput.addEventListener("change", (event) => handleImageUpload(event, imageProcessor));
});

async function checkServerStatus() { // Check if the server is running
  const statusDiv = document.getElementById("server-status")
  if (!statusDiv) return

  try {
    const response = await fetch("http://localhost:3000/test")
    if (response.ok) {
      statusDiv.textContent = "Server Status: Connected ✓"
      statusDiv.className = "status-ok"
    } else {
      statusDiv.textContent = "Server Status: Error - Server returned " + response.status
      statusDiv.className = "status-error"
    }
  } catch (error) {
    statusDiv.textContent =
    "Server Status: Error - Cannot connect to server. Make sure it's running on http://localhost:3000 by running 'node server.js' in the terminal" 
    statusDiv.className = "status-error"
    console.error("Server connection error:", error)
  }
}

document.addEventListener("DOMContentLoaded", checkServerStatus)

resultsDiv.innerText = "Select an image file to process."

export { metadata, predictions };
