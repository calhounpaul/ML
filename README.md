# ML Repository

This repository is a comprehensive toolkit for advanced machine learning and AI tasks, focusing on natural language processing, image processing, model fine-tuning, and various AI-powered applications. It provides a well-organized structure of scripts, tools, and configurations to support various aspects of machine learning workflows.

## Key Features

- **Environment Setup**: Scripts for setting up CUDA, Docker, and other prerequisites.
- **Model Implementations**: Support for text generation (GPT-like models), image processing (Stable Diffusion, Segment Anything Model), and more.
- **Fine-tuning Tools**: Scripts and configurations for fine-tuning language models.
- **Containerization**: Extensive use of Docker for reproducible and portable ML environments.
- **Integration**: Works with popular platforms like Hugging Face and frameworks such as PyTorch and Transformers.
- **Full Pipeline Support**: Includes tools for data processing, model training, and inference.
- **Text-to-Speech**: Integration with TTS (Text-to-Speech) web UI.
- **Image Generation and Manipulation**: Scripts for running Stable Diffusion and ComfyUI.
- **Web Scraping**: Selenium-based tools for web data collection.
- **E-book Processing**: Tools for processing and analyzing e-books.

## Repository Structure

- `permathings/`: Core directory containing libraries, prerequisites, and scripts.
  - `libs/`: Utility libraries for various tasks.
  - `prereqs/`: Scripts for setting up the environment (CUDA, Docker, etc.).
  - `scripts/`: Various scripts for different ML tasks and model implementations.
- `README.md`: This file.
- `.gitignore`: Git ignore file.

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/calhounpaul/ML.git
   ```

2. Set up the environment (this is currently designed to run on an ub24.04 VM in XCP-ng, but it should also work with any cloud provider):
   ```
   cd ML/permathings/prereqs
   bash ./all.sh
   ```

3. Initialize secrets:
   ```
   cd ../scripts/utils
   bash ./init_secrets.sh
   ```

4. Choose a specific task or model from the `scripts/` directory and follow the instructions in the respective script or README.

## Key Components

- **Text Generation**: Includes scripts for running text generation web UIs and VLLM servers.
- **Image Processing**: Tools for Stable Diffusion, Segment Anything Model (SAM), and other image segmentation tasks.
- **LLM Integration**: Scripts for running various Large Language Models, including LLaVA and OpenDevin.
- **Fine-tuning**: Tools for fine-tuning language models with different techniques.
- **Data Processing**: Scripts for e-book processing and dataset creation.
- **Web Interfaces**: Docker configurations for running web-based interfaces for various AI tasks.

## Requirements

- CUDA-compatible GPU
- Ubuntu 24.04