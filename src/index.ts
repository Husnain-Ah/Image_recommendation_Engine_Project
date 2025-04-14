import * as tf from "@tensorflow/tfjs"
import * as mobilenet from "@tensorflow-models/mobilenet"
import { searchLocalImages } from "./imageSearch";

const imageInput = document.getElementById("image-input")! as HTMLInputElement
const imageDisplay = document.getElementById("image-display")! as HTMLImageElement
const resultsDiv = document.getElementById("results")!
const loadingDiv = document.getElementById("loading")!
const imageGallery = document.getElementById("image-gallery")!

let userPreferenceVector: tf.Tensor | null = null;
let currentImageEmbedding: tf.Tensor | null = null;


let model: mobilenet.MobileNet | null = null

async function initializeBackend() {
  try {
    await tf.setBackend("webgl")
    await tf.ready()
    console.log("TensorFlow.js backend initialized:", tf.getBackend())
  } catch (error) {
    console.error("Error initializing TensorFlow.js backend:", error)
    resultsDiv.innerText = "Error initializing TensorFlow.js backend."
  }
}

async function loadModel() {
  console.log("Loading MobileNet model...")
  loadingDiv.style.display = "block"

  try {
    model = await mobilenet.load()
    console.log("Model loaded successfully.")
    resultsDiv.innerText = "Model loaded. Ready for image selection."
  } catch (error) {
    console.error("Error loading model:", error)
    resultsDiv.innerText = "Error loading model. Please try again later."
  } finally {
    loadingDiv.style.display = "none"
  }
}

function getCosineSimilarity(tensorA: tf.Tensor, tensorB: tf.Tensor): number {
  const dotProduct = tf.sum(tf.mul(tensorA, tensorB));
  const normA = tf.norm(tensorA);
  const normB = tf.norm(tensorB);
  const similarity = dotProduct.div(normA.mul(normB));
  return similarity.dataSync()[0];
}

async function handleImageUpload(event: Event) {
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
      console.log("Processing image...")

      if (!model) {
        console.log("Loading model...")
        model = await mobilenet.load()
      }

      const tensor = tf.browser.fromPixels(imageDisplay)
      console.log("Image tensor shape:", tensor.shape)

      if (tensor.shape[0] === 0 || tensor.shape[1] === 0) {
        resultsDiv.innerText = "Invalid image dimensions."
        loadingDiv.style.display = "none"
        return
      }

      console.log("Classifying image...")
      const predictions = await model.classify(tensor)
      console.log("Classification results:", predictions)
      displayResults(predictions)

      if (predictions && predictions.length > 0) {
        const topPrediction = predictions[0].className.split(",")[0].trim()
        console.log("Top prediction:", topPrediction)
        console.log("Searching for images related to:", topPrediction)

        const embeddingModel = model as any;
        currentImageEmbedding = embeddingModel.infer(tensor, true) as tf.Tensor;


        imageGallery.innerHTML = "<p>Searching for related images...</p>"

        const imageUrls = await searchLocalImages(topPrediction);
        console.log("Received image URLs:", imageUrls)
        displayImageResults(imageUrls)
      }
    } catch (error) {
      console.error("Error during classification:", error)
      resultsDiv.innerText = "Error processing image: " + (error as Error).message
    } finally {
      loadingDiv.style.display = "none"
    }
  }
}

function displayResults(predictions: any) {
  if (!predictions || predictions.length === 0) {
    resultsDiv.innerText = "No classification results."
    return
  }

  let resultText = "Predictions:<br><ul>"
  predictions.forEach((prediction: { className: string; probability: number }) => {
    resultText += `<li>${prediction.className}: ${(prediction.probability * 100).toFixed(2)}%</li>`
  })
  resultText += "</ul>"
  resultsDiv.innerHTML = resultText
}

const BASE_URL = "http://localhost:3000";
const SIMILARITY_THRESHOLD = 0.3;

async function displayImageResults(imageUrls: string[]) {
  imageGallery.innerHTML = "";

  if (!imageUrls || imageUrls.length === 0) {
    imageGallery.innerHTML = "<p>No related images found</p>";
    return;
  }

  const imageScores: { url: string; score: number }[] = [];

  for (const url of imageUrls) {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `${BASE_URL}/${url}`;

    const loadPromise = new Promise<HTMLImageElement>((resolve, reject) => {
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${img.src}`));
    });

    try {
      const loadedImg = await loadPromise;
      const tensor = tf.browser.fromPixels(loadedImg);

      const embeddingModel = model as any;
      const embedding = embeddingModel.infer(tensor, true) as tf.Tensor;

      let score = 0;
      if (userPreferenceVector) {
        score = getCosineSimilarity(embedding, userPreferenceVector);
      }

      if (score >= SIMILARITY_THRESHOLD || !userPreferenceVector) {
        imageScores.push({ url, score });
      }
    } catch (err) {
      console.error(err);
    }
  }

  imageScores.sort((a, b) => b.score - a.score);

  for (const { url, score } of imageScores) {
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
  console.log("Final filtered image URLs:", imageScores.map(i => i.url));

  showRatingSection();
}


function showRatingSection() {
  const ratingSection = document.getElementById('rating-section');
  if (ratingSection) {
    ratingSection.style.display = 'block';
  }
}

document.getElementById('submit-rating')!.addEventListener('click', async () => {
  const rating = parseInt((document.getElementById('rating-input') as HTMLInputElement).value);

  if (rating >= 1 && rating <= 10) {
    console.log('User rating:', rating);

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

    alert('Thank you for your rating!');
  } else {
    alert('Please enter a rating between 1 and 10');
  }
});

document.getElementById("reset-preferences")!.addEventListener("click", () => {
  if (userPreferenceVector) {
    userPreferenceVector.dispose();
    userPreferenceVector = null;
  }
  alert("User preference vector has been reset.");
});


initializeBackend().then(() => {
  loadModel().then(() => {
    console.log("Model loaded and ready")
  })

  imageInput.addEventListener("change", handleImageUpload)
})

async function checkServerStatus() {
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
