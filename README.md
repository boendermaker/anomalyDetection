# Context-Aware Motor Anomaly Detection

This repository contains a **Contextual Anomaly Detection** system for industrial motor monitoring. Unlike traditional static threshold monitors, this AI-driven solution distinguishes between expected high-current events (like start-up sequences) and actual mechanical faults by analyzing amperage in relation to its operational timeframe (`step`).

## 🚀 The Core Concept

In industrial environments, a motor's "Normal" state changes over time. A 20A current is perfectly fine during the first 5 seconds (Start-up), but the same 20A could indicate a critical failure if it occurs during steady-state operation.

This project uses a Neural Network to learn the **Decision Boundary** between:
1. **Current (`current`)**: The measured load.
2. **Sequence/Time (`step`)**: The relative position within the cycle.

By training on both features, the model filters out "False Positives" during the start-up phase automatically.

---

## 🛠 Features

* **Dynamic Synthetic Data**: Generates realistic motor curves including start-up (exponential decay), steady-state, and coast-down phases with randomized phase lengths.
* **Data is Prelabeled**: You can modify the generation of data with prelabeled Datapoints in DataGenerator Class
* **On-the-Fly Training**: Trains a 3-layer Neural Network directly in the Node.js backend using TensorFlow.js.
* **Live Inference**: Visualizes probability overlays (heatmaps) over the motor curve in real-time.

---

## 🏗 Architecture & Implementation Note

**Disclaimer:** This is a **Baseline Implementation** intended for educational purposes and proof-of-concepts. 

### Scalability & Continuous Monitoring
The architecture is designed to be highly adaptable for professional environments:
* **Long-Term Time Series**: Can be scaled to handle thousands of data points (e.g., monitoring a motor over an entire shift).
* **Sliding Window Monitoring**: The system can be adapted for **continuous live-frames**, where the AI constantly analyzes a rolling buffer of the last "N" seconds.
* **Feature Engineering**: For industrial use, additional features like "Moving Average" or "Standard Deviation" can be added to the input layer to improve stability.

---

## 🐳 Getting Started (Docker Compose)

The easiest way to run this project is using Docker. This ensures all C++ bindings for TensorFlow are correctly compiled.

### 1. Prerequisites
* Install [Docker](https://docs.docker.com/get-docker/)
* Install [Docker Compose](https://docs.docker.com/compose/install/)

### 2. Running the Application
Clone the repo and run:

```bash
docker compose up -d --build
docker compose restart tf-app (When you made changes)

Locate your browser to http://localhost:3000