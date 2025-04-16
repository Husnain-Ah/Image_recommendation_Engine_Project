import * as d3 from 'd3';
import * as tf from "@tensorflow/tfjs"

import { predictions, metadata } from './index';


const BASE_URL = "http://localhost:3000";

type ImageNode = {
  id: string;
  url: string;
  label?: string;
  score?: number;
  isMain?: boolean;
};

type BarChartData = {
  index: number;
  value: number;
};

/// Function to render the force-directed graph to show the 15 most similar images to the uploaded image
export function renderForceGraph(uploadedImageUrl: string, topImages: { url: string; score: number }[]) {
  const container = d3.select("#force-graph");
  container.selectAll("*").remove(); 

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
    .attr("dy", 5) 
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
    .attr("stroke", (d: { target: string | { id: string } }) => {
      const targetId = typeof d.target === "string" ? d.target : d.target.id;
      const targetNode = nodes.find(n => n.id === targetId);
      if (targetNode?.label === nodes[0].label) return "green";
      return "red";
    })    
    .attr("stroke-width", d => 2 * d.weight);

    const node = svg.selectAll("image")
      .data(nodes)
      .enter()
      .append("image")
      .attr("xlink:href", d => d.url)
      .attr("width", 40)
      .attr("height", 40)
      .attr("class", "force-node")
      .style("stroke", d => {
        if (d.isMain) return "black";
        if (d.label === nodes[0].label) return "green";
        return "blue";
      })
      .style("stroke-width", 3)
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

// Function to render the user vector chart to show the top 10 dimensions in the user preference profile
// based on the images the user has rated. Each bar represents how strongly the user favors a certain visual feature
export function renderUserVectorChart(userVector: tf.Tensor) {
  const container = d3.select("#user-vector-chart");
  container.selectAll("*").remove();

  if (!userVector) { //doesnt work
    container.append("p")
      .text("No preference data yet. Start rating some images!")
      .style("color", "#888")
      .style("font-style", "italic")
      .style("text-align", "center")
      .style("padding", "1rem");
    return;
  }

  const vectorData: BarChartData[] = Array.from(userVector.dataSync()).map((value, index) => ({
    index,
    value
  }));

  // Top 10 dimensions with highest weights
  const topDims = vectorData.sort((a, b) => b.value - a.value).slice(0, 10);

  const width = 500;
  const height = 300;
  const margin = { top: 40, right: 20, bottom: 50, left: 60 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  const x = d3.scaleBand()
    .domain(topDims.map(d => d.index.toString()))
    .range([0, chartWidth])
    .padding(0.1);

  const y = d3.scaleLinear()
    .domain([0, d3.max(topDims, d => d.value)!])
    .nice()
    .range([chartHeight, 0]);

  const chart = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  chart.selectAll("rect")
    .data(topDims)
    .enter()
    .append("rect")
    .attr("x", d => x(d.index.toString())!)
    .attr("y", d => y(d.value))
    .attr("width", x.bandwidth())
    .attr("height", d => chartHeight - y(d.value))
    .attr("fill", "#3f37c9")
    .append("title")
    .text(d => `Dimension ${d.index}: Weight = ${d.value.toFixed(2)}`);

  // Labels for the bars
  chart.selectAll("text")
    .data(topDims)
    .enter()
    .append("text")
    .attr("x", d => x(d.index.toString())! + x.bandwidth() / 2)
    .attr("y", d => y(d.value) - 5)
    .attr("text-anchor", "middle")
    .text(d => d.value.toFixed(2))
    .style("font-size", "12px");

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", 20)
    .style("text-anchor", "middle")
    .style("font-size", "16px")
    .text("Top User Preference Dimensions");

  // Explanatory paragraph
  container.append("p")
    .attr("class", "user-vector-description")
    .style("max-width", "500px")
    .style("margin", "0.5rem auto")
    .style("text-align", "center")
    .style("font-size", "14px")
    .style("color", "#555")
    .text("This chart shows the top 10 dimensions in your preference profile based on the images you've rated. Each bar represents how strongly you favor a certain visual feature, extracted from deep image embeddings. The higher the value, the more influence that feature has on your future recommendations.");

  // Y-axis label
  svg.append("text")
    .attr("transform", `rotate(-90)`)
    .attr("y", margin.left / 3)
    .attr("x", -height / 2)
    .attr("text-anchor", "middle")
    .style("font-size", "12px")
    .text("Preference Weight");

  // X-axis label
  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 10)
    .attr("text-anchor", "middle")
    .style("font-size", "12px")
    .text("Embedding Dimension Index");
}
