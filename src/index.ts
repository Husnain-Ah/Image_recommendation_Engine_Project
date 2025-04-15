import * as tf from "@tensorflow/tfjs"
import * as mobilenet from "@tensorflow-models/mobilenet"
import { searchLocalImages } from "./imageSearch";
import * as d3 from 'd3';

const imageInput = document.getElementById("image-input")! as HTMLInputElement
const imageDisplay = document.getElementById("image-display")! as HTMLImageElement
const resultsDiv = document.getElementById("results")!
const loadingDiv = document.getElementById("loading")!
const imageGallery = document.getElementById("image-gallery")!

let userPreferenceVector: tf.Tensor | null = null;
let currentImageEmbedding: tf.Tensor | null = null;
let predictions: any[] = [];

let model: mobilenet.MobileNet | null = null

//used for forcegraph
type ImageNode = {
  id: string;
  url: string;
  label?: string;
  score?: number;
  isMain?: boolean;
};

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
  console.log("Loading MobileNetV2 model...")
  loadingDiv.style.display = "block"

  try {
    model = await mobilenet.load({ version: 2, alpha: 1.0 }); //  MobileNetV2 too match annoy feature extraction
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
      predictions = await model.classify(tensor)
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
let metadata: Record<string, { label: string }> = {};

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

function getKeywordScore(labelA: string, labelB: string): number {
  if (!labelA || !labelB) return 0;
  const aWords = labelA.split(/[ ,]+/);
  const bWords = labelB.split(/[ ,]+/);
  const shared = aWords.filter(word => bWords.includes(word));
  return shared.length > 0 ? 1 : 0;
}

async function displayImageResults(imageUrls: string[]) {
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

      if (userPreferenceVector && currentImageEmbedding) {
        // Hybrid scoring: combine user preferences and content-similarity, this is 70%user preference and 30% current image, user preference matters more
        const similarityWithPreference = getCosineSimilarity(embedding, userPreferenceVector);
        const similarityWithCurrentImage = getCosineSimilarity(embedding, currentImageEmbedding);

        // Weighted hybrid score (tweak weights as needed), 
        score = 0.7 * similarityWithPreference + 0.3 * similarityWithCurrentImage;
      } else if (userPreferenceVector) {
        // Use only user preference, 
        score = getCosineSimilarity(embedding, userPreferenceVector);
      } else if (currentImageEmbedding) {
        // Use only current image embedding, mainly used at start when there is no user preference determined by rating
        score = getCosineSimilarity(embedding, currentImageEmbedding);
      } else {
        // Fallback (shouldnt hit)
        score = 0;
      }


      const labelB = metadata[filename]?.label ?? "";
      const keywordScore = getKeywordScore(uploadedLabel, labelB);
      const finalScore = 0.8 * score + 0.2 * keywordScore;

      imageScores.push({ url, score: finalScore });
    } catch (err) {
      console.error(err);
    } 
  }

  spinner.style.display = "none";

  const SIMILARITY_THRESHOLD = 0.1; //lower threshold to get more images, increase later to get more strict image filtering

  // top k filtering gives top 5 images after filtering by similarity threshold
  const k = 5; // Number of top images to display

  const topImages = imageScores
    .filter(img => img.score >= SIMILARITY_THRESHOLD)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);

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

  showRatingSection();
  renderScoreChart(topImages);
  renderForceGraph(imageDisplay.src, topImages);
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

function renderForceGraph(uploadedImageUrl: string, topImages: { url: string; score: number }[]) {
  const container = d3.select("#force-graph");
  container.selectAll("*").remove(); // Clear previous graph

  const width = 600;
  const height = 400;

  const nodes: ImageNode[] = [
    { id: "uploaded", url: uploadedImageUrl, isMain: true, label: predictions[0]?.className },
    ...topImages.map((img, i) => {
      const filename = img.url.split("/").pop()!;
      const label = metadata[filename]?.label ?? "Unknown";
      return {
        id: `img${i}`,
        url: `${BASE_URL}/${img.url}`,
        score: img.score,
        label
      };
    })
  ];
  
  const links = topImages.map((img, i) => ({
    source: "uploaded",
    target: `img${i}`,
    weight: img.score
  }));

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  const labels = svg.selectAll("text")
    .data(nodes)
    .enter()
    .append("text")
    .attr("class", "node-label")
    .attr("text-anchor", "middle")
    .attr("dy", 55) 
    .style("font-size", "10px")
    .text(d => d.label || "");

  const simulation = d3.forceSimulation(nodes as any)
    .force("link", d3.forceLink(links).id((d: any) => d.id).distance(d => 250 * (1 - (d as any).weight)))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2));

  const link = svg.selectAll("line")
    .data(links)
    .enter()
    .append("line")
    .attr("stroke", "#aaa")
    .attr("stroke-width", d => 2 * d.weight);

    const node = svg.selectAll("image")
      .data(nodes)
      .enter()
      .append("image")
      .attr("xlink:href", d => d.url)
      .attr("width", 40)
      .attr("height", 40)
      .attr("class", "force-node")
      .call(
        d3.drag<SVGImageElement, ImageNode>()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      );

      simulation.on("tick", () => {
        link
          .attr("x1", d => (d.source as any).x)
          .attr("y1", d => (d.source as any).y)
          .attr("x2", d => (d.target as any).x)
          .attr("y2", d => (d.target as any).y);
      
        node
          .attr("x", (d: any) => d.x - 20)
          .attr("y", (d: any) => d.y - 20);
      
        labels
          .attr("x", (d: any) => d.x)
          .attr("y", (d: any) => d.y + 30); 
      });

  function dragstarted(event: any, d: any) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event: any, d: any) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event: any, d: any) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
}

function renderScoreChart(images: { url: string; score: number }[]) {
  const container = d3.select("#score-chart");
  container.selectAll("*").remove();

  const width = 500;
  const height = images.length * 40 + 60;

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  const margin = { top: 40, right: 60, bottom: 50, left: 60 }; 
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const x = d3.scaleLinear()
    .domain([0, d3.max(images, d => d.score) || 0])
    .range([0, chartWidth]);

  const y = d3.scaleBand()
    .domain(images.map(d => d.url))
    .range([0, chartHeight])
    .padding(0.1);

  const chart = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  chart.selectAll("rect")
    .data(images)
    .enter()
    .append("rect")
    .attr("x", 0)
    .attr("y", d => y(d.url) || 0)
    .attr("width", d => x(d.score) || 0)
    .attr("height", y.bandwidth() || 0)
    .attr("fill", "#69b3a2");

  chart.selectAll("image.thumbs")
    .data(images)
    .enter()
    .append("image")
    .attr("x", -50) 
    .attr("y", d => y(d.url) || 0)
    .attr("width", 40)
    .attr("height", y.bandwidth() || 0)
    .attr("href", d => `${BASE_URL}/${d.url}`);

  chart.selectAll("text.scores")
    .data(images)
    .enter()
    .append("text")
    .attr("x", d => (x(d.score) || 0) + 5) 
    .attr("y", d => (y(d.url) || 0) + (y.bandwidth() || 0) / 2)
    .attr("dy", ".35em")
    .attr("text-anchor", "start")
    .text(d => d.score.toFixed(2))
    .style("font-size", "12px")
    .style("fill", "#333");

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 10)
    .style("text-anchor", "middle")
    .text("Similarity Score");

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", 20)
    .style("text-anchor", "middle")
    .style("font-size", "16px")
    .text("How similar these images are to your uploaded image");
}
  
document.addEventListener("DOMContentLoaded", checkServerStatus)

resultsDiv.innerText = "Select an image file to process."
