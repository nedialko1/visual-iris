# Visual-Iris 👁️ A Journey into Explainable & Frugal AI

## Table of Contents
* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Installation and Setup](#installation-and-setup)
* [Quick Introduction](#quick-introduction)
* [Repository Elements Purpose and Use Details](#repository-elements-purpose-and-use-details)
* [Post Scriptum](#post-scriptum)

---

## Project Overview

This little project provides the companion set of python codes of an article on explainable AI (XAI).
This codes target eye-centric didactic purposes with visually enriched user experience being the topmost priority.

> *Here a rather famous dataset is used inside a glass-box full of hands-on labs toward a visual introduction to the mechanics of machine learning (ML). 
Such better understanding may contribute to more out-of-the-box ML model optimization ideas.*

---

### 📚 Companion Resource 
This repository is designed to be explored alongside the Computer Science article: **Frugal Explainable AI/ML Model Optimization - a Hands-On Introduction**. The latter text provides the theoretical depth and statistical proofs behind the parsimonious architectures demonstrated here, serving as a comprehensive guide for those seeking to move beyond the "black-box/black-windshield" modeling and toward genuinely explainable ML engineering. 

---

### 📈 Summary
Structural optimization is at the intersection of mathematical clarity and computational efficiency. The cases presented visually here argue that explainability is nothing less of a key requirement for robust AI design.

* For more insights and depth, refer to the companion article: **Frugal Explainable AI/ML Model Optimization
a Hands-On Introduction**

---

## Repository Structure 

**  REPO LAYOUT FOR THE QUICK DIVER: **

```
visual_iris/
├── levels/               # The "Article!" Logic (scripts beyond the simplest demos)
│   ├── start_here        # 'classic' iris MLP (4-8-9-3) architecture using PyTorch.
│   ├── da_Visual_Iris    # the Visual Iris GUI: 
│   │         # made to ingest various simple model  flavors
│   │         # Sliders “light” or “shut” individual neurons
│   ├── da_Statistical_Iris   # Factored segmentation logic
│   ├── da_Pretrained_Iris    # Architectural Optimization: 
│   │           # The 12-Parameter Frugality Challenge
│   ├── Model_Drift      # Model Drift Analysis Visualization
│   └── PWL_tool       # the ReLU Interactive Definition GUI
│
├── etc_utils/             # The useful "Tools" (key Deduplicated Classes & helpers)
│   ├── viz_bloks   # Building blocks for impactful visuals and GUI’s
│   │        # e.g. the Hinton diagrams of model weights - 
│   │        #  for a given model version or for versions comparison   
│   │ 
│   └── data_harvest_io     # Get Train/Test data: URL &  Path handling
│
├── models_io/      # NN Models & parameters loading/saving; pt/YAML/JSON etc
│
├── demos/             # The Impactful Demos (readily runnable scripts)
│   ├── test_me_1.py   # The baseline iris MLP (4-8-9-3) architecture
│   ├── test_me_2.py   # The Model Drift Analysis Visualization
│   └── test_me_3.py   # Using the levels/PWL_tool/PWL_bounds.py (* 2)
│
└── images/            # mostly saved plots (* 1)
```

Notes: 
(* 1) Representative examples and figures provide a visual intuition as to what to expect from these approaches.

(* 2) Amongst other uses, the **demos** folder scripts can be used toward checking the installation and also to introduce some of the tools of interest.

---

**How to use toward best results**

The repository is organized in the following way:

* Its **root folder** contains all you need to get started - e.g. the minimum *requirements* file 

* The **demos** folder contains fully functional scripts that implement and run 
(do the inference, training etc) the well-known **baseline** version of
the *Iris Flowers* classifier MLP *PyTorch* network model. 

* The same model is then examined under the magnifying glass of
progressively more elaborate approaches toward the model's better understanding and refinement.   
These approaches have dedicated subfolders as described below.

---

## Installation and Setup 🛠️ 

1. **Repository Initialization:**

```bash
git clone https://github.com/nedialko1/visual-iris.git
cd visual-iris
```

---

2. **Dependency Installation:**

```bash

pip install -r requirements.txt
```

CUDA Configuration: If utilizing a GPU, verify your version in requirements.txt. 
The CUDA version assumed is 12.4.

---

## Quick Introduction

This repository provides an introductory framework for exploring connectionist models through the lens of Explainable AI (XAI) and structural optimization. It aims to bridge the gap between high-level neural network performance and intuitive human understanding.
The repository is organized around a technical progression from the 'classic' neural network implementation baselines toward optimized and statistically grounded architectures. 
<br>
The work is divided into four stage Levels:

---

* **Level 1: Baseline Implementation** – Establishes a benchmark using a 4-8-9-3 Multi-Layer Perceptron (MLP). This stage covers data acquisition, preprocessing, and the fundamental mechanics of connectionist training.

* **Level 2: Interpretation through Visualization** – Focuses on the visual unfolding of hidden layer activity. This section addresses Explainable AI (XAI) through the observation of decision boundaries and unit activations.

* **Level 3: Statistical Grounding** – Benchmarks the neural network against Principal Component Analysis (PCA) and Fisher Linear Discrimination (FLD). It evaluates whether connectionist complexity is required or whether closed-form statistical solutions offer alternatives and comparable performance.

* **Level 4: Architecture Optimization** – Demonstrates parameter reduction by utilizing statistical priors to pre-train specific layers. The process transitions from a 151-parameter model to a 12-parameter output mapping while maintaining the classification accuracy.

---

## Repository Elements Purpose and Use Details

Foremost let us have a look at the standard iris data set. This can be readily obtained for the UCI page
`https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`

![Fig. 1.0 - UCI Data Loading](./images/load_uci_data.png)

### Begin your exploration in the `/demos` folder

![Fig. 1.0.1 - The **demos** and their place in the repo ](./images/GIT_2_inventory_intro.png)

The python scripts in this folder are meant also to verify that
Installation and Setup](#installation-and-setup) was successfully completed.

* try_me = almost certainly the very first script to try; running it should also produce 
  the following figure:

  ![Fig. 1.0.1.A - The NN model's UN-trained performance ](./images/baseline_MLP_UN_trained_performance.png)

* test_me_1 = The previous demo illustrates the untrained performance of the model described 
  in more detail by the next section. In addition , *this* demo does a model training run and then 
  illustrates the time course of performance evolution during training:

 ![Fig. 1.0.1.B - The NN model's performance evolution during training ](./images/MLP_train_performance_evolution.png)   

 In conclusion, the demo illustrates the trained performance of the model:

 ![Fig. 1.0.1.C - The NN model's trained performance ](./images/baseline_MLP_classification_trained.png) 

### 🟢 Level 1: Iris Flowers Classifier Neural Network (`/start_here`)
*the 'classic' baseline Multi-Layer Perceptron (MLP) model*

Introduction to the 'standard' MLP (4-8-9-3) architecture using PyTorch.

* **Goal:** Establish a performance benchmark.
* **Action:** Execute `python baseline_classifier.py` from the root directory.
* **Metrics:** The script initializes 151 parameters. This baseline serves as the control group for all subsequent optimizations experiments.

Here the repository offers 3 parameter sets of pre-computed medel weights & biases. 
These are stored in the subfolder `./start_here/rand_weights`:

* when the weights are initialized at random, the NN classifier has quality lower than chance - 
e.g. in the following scatter plot all samples are classified as Setosa (100 out of 150 incorrectly!)

![Fig. 1.1.A - - UCI Data Classification - untrained](./images/baseline_MLP_classification_UN_trained.png)

* even when the weights are trained, errors are possible as the data points of the different classes
  form clusters having significant spatial overlap   

![Fig. 1.1.B - UCI Data Classification - trained](./images/baseline_MLP_classification_trained.png)

> 🔲 **To reproduce Fig. 1.1**
> 
> To reproduce the dataset distribution shown in **[Fig. 1.1]**, please execute the following source script:
> 
> ```bash
> python load_uci_data.py
> ```

---

### 🟡 Level 2: Interpretability & Visualization (`/da_Visual_Iris`)
*Open the Black Box: Visualize Model Architecture and Activation Patterns*

This section focuses on the visual unfolding of the connectionist model to achieve Explainable AI (XAI).

* **Action:** Navigate `cd da_Visual_Iris` and execute `python visual_iris_main.py`.
* **Interaction:** Manipulate the layer-specific sliders to observe how thresholding affects decision boundaries in real-time.

* **Key Feature: Activation Thresholding.** Interactive controls allow for the manual setting of unit firing thresholds.
* **Objective:** Observe the potential transition from continuous rate coding to sparse activation patterns, exploring the prerequisites for Spiking Neural Networks (SNN).

+++
![Setosa Baseline](./images/visual_iris/Setosa_Baseline.png) 
![Versicolor Baseline](./images/visual_iris/Versicolor_Baseline.png) 
![Virginica Baseline](./images/visual_iris/Virginica_Baseline.png)

---

### 🟠 Level 3: Statistical Grounding (`/da_Statistical_Iris`)
*Comparative Analysis: Neural Networks vs Statistics-powered Parsimony*

An juxtaposition of connectionist learning against classical Principal Component Analysis (PCA) and Fisher Linear Discrimination (FLD).

* **Action:** Navigate `cd Statistical_Iris` and execute `python GUI_data.py`.
* **Interaction:** Manipulate the mouse to define PWL decision boundaries.

* **Key Finding:** Classical statistical methods can achieve robust discrimination using closed-form solutions with fewer parameters.
* **Structural Observation:** We visualize instances where over-parameterized models achieve accuracy despite the hidden layers failing to represent the underlying problem geometry.

![Stats 2D](./images/stats_iris_2D.png) 

![Stats Take 4 F1](./images/stats_iris_2D_take4_F1.png) 
![Stats Take 4 F2](./images/stats_iris_2D_take4_F2.png)

![Setosa Refined 1](./images/visual_iris/Setosa_Refined_1.png) 
![Versicolor Refined 1](./images/visual_iris/Versicolor_Refined_1.png) 
![Virginica Refined 1](./images/visual_iris/Virginica_Refined_1.png)

---   

### 🔴 Level 4: Architectural Optimization (`/da_Pretrained_Iris`)
*Frugality: The 15-Parameter Challenge*

![Setosa Refined 2](./images/visual_iris/Setosa_Refined_2.png) 
![Versicolor Refined 2](./images/visual_iris/Versicolor_Refined_2.png) 
![Virginica Refined 2](./images/visual_iris/Virginica_Refined_2.png)

![LDA 3 classes --> 2D](./images/LDA_01.png)

![Setosa Refined 2](./images/visual_iris/Setosa_Refined_3.png) 
![Versicolor Refined 2](./images/visual_iris/Versicolor_Refined_3.png) 
![Virginica Refined 2](./images/visual_iris/Virginica_Refined_3.png)

---

### 🔵 Level 5: The PWL tool (`/PWL_tool`)
*The Interactive ReLU Designer in 2D*

* demos/test_me_3.py --> levels/PWL_tool/PWL_bounds.py

![Toy ReLU Constraints](./images/PWL_Designer/Toy_ReLUs_Constraints.png)

* levels/PWL_tool/PWL_tool_V2.py

A pretrained NN model fully determined by mathematically precise methods and generated interactively:

![ReLU units by PCA & FLD](./images/PWL_Designer/PCA_FLD_ReLUs.png)

---

### 🟣 Level 6: The Model Drift Lesson (`/Model_Drift`)
*Implicit Frugality embedded in the higher-dimensional model parameter space*

This part demonstrates the reduction of model complexity through informed architectural design.

* **Action:** Run `python extreme_optimizer.py`.
* **Objective:** Demonstrate that informed initialization based on statistical priors significantly reduces the training overhead and parameter count.

* **Key Finding:** 
  The **Architectural Evolution:**
  - 1. **Baseline:** 151 parameters.
  - 2. **Compressed:** 48 parameters (4-4-3-3).
  - 3. **Optimized:** 36 [actually 12] parameters (4-3-3-3), utilizing fixed PCA/FLD weights.
* **Structural Observation:** The 12-Parameter Performance: Achieving **97.34% accuracy** by training solely the output mapping layer.

![netron view of baseline MLP](./images/tensors_heatmaps/netron_baseline_MLP.png)

![Model Drift (using Hinton diagrams)](./images/hinton_drift.png)

![Baseline MLP Model Drift (Heatmap)](./images/tensors_heatmaps/architecture_drift.png)

--- 

## Post Scriptum 🏛️

### The "Russian Nested Dolls" Approach to Observability 
The transition from the minimalist Iris dataset to production-scale may be managed through an abstraction reminiscent  
of the *Russian Nested Dolls*. To preserve observability in more complex systems the model may viewed as a nested structure: individual neurons reside within layers, layers within modules, and modules within subsystems. By the visualization of the "innermost doll"—the basic unit of the Iris classifier—one develops the intuition to zoom out and interpret behaviors at larger-scales and higher dimension systems staying in touch with the key moving parts. 

### On AI Agency and the Human Architect 
The implementation of this repository reflects a contemporary collaboration between human intent and AI agency. While the source code was generated through the lens of modern AI tools, the underlying engineering strategies, mathematical derivations, and optimization logic remain 100% human-original. In this paradigm, the AI acts as a high-fidelity technical executor—a "digital compiler" for human-prescribed recipe—ensuring that the final code adheres to rigorous, design requirements while the architect remains at the source of the key algorithmic features. 


