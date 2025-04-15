import * as d3 from 'd3';

import { predictions, metadata } from './index';


const BASE_URL = "http://localhost:3000";

type ImageNode = {
  id: string;
  url: string;
  label?: string;
  score?: number;
  isMain?: boolean;
};



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
    .attr("stroke", d => {
      const targetNode = nodes.find(n => n.id === d.target);
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

export function renderScoreChart(images: { url: string; score: number }[]) {
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