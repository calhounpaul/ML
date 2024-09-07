# ML Repository

This repository is a comprehensive toolkit for advanced machine learning and AI tasks, focusing on natural language processing, image processing, and model fine-tuning. It provides a well-organized structure of scripts, tools, and configurations to support various aspects of machine learning workflows.

## Key Features

- **Environment Setup**: Scripts for setting up CUDA, Docker, and other prerequisites.
- **Model Implementations**: Support for text generation (GPT-like models), image processing (Stable Diffusion, Segment Anything Model), and more.
- **Fine-tuning Tools**: Scripts and configurations for fine-tuning language models.
- **Containerization**: Extensive use of Docker for reproducible and portable ML environments.
- **Integration**: Works with popular platforms like Hugging Face and frameworks such as PyTorch and Transformers.
- **Full Pipeline Support**: Includes tools for data processing, model training, and inference.

## Repository Structure

- `permathings/`: Main directory containing scripts and tools
  - `libs/`: Utility libraries (e.g., docker_tools, secretary, selenium_tools)
  - `prereqs/`: Scripts for setting up the environment (CUDA, Docker, etc.)
  - `scripts/`: Various ML model implementations and tools
    - `ebook_processing/`: Tools for processing and analyzing ebooks
    - `finetuning/`: Scripts for fine-tuning language models
    - `langchain/`: Integration with LangChain framework
    - `ollama/`: Scripts for running Ollama models
    - `segmentation/`: Image segmentation tools (SAM, OpenAdapt, etc.)
    - `stable_diffusion/`: Scripts for running Stable Diffusion
    - `textgen_webui/`: Text generation web UI setup
    - `tts_webui/`: Text-to-Speech web UI setup
    - `utils/`: Utility scripts (e.g., git repository analysis, secret management)
    - `vllm/`: Scripts for running vLLM models
- `.gitignore`: Git ignore file
- `README.md`: This file

## Getting Started

Note: The packages in this repo are designed to be compatible with fresh Ubuntu 24.04 VMs spawned on a home Xen server. Use caution when deploying it anywhere else.

1. Clone the repository:
   ```
   git clone https://github.com/calhounpaul/ML.git
   ```

2. Set up the environment:
   ```
   cd ML/permathings/prereqs
   bash ./all.sh
   ```

3. Initialize secrets (just HF token at the moment):
   ```
   cd ../scripts/utils
   bash ./init_secrets.sh
   ```

4. Choose a specific task or model from the `scripts/` directory and run it.

## Requirements

- CUDA-compatible GPU
- Ubuntu 24.04

## Additional Notes

- The repository includes scripts for various AI tasks, including text generation, image processing, and text-to-speech.
- There are tools for working with ebooks, fine-tuning models, and integrating with frameworks like LangChain.
- The project makes extensive use of Docker for containerization and reproducibility.
- Many scripts are provided for setting up and managing the environment, including CUDA and Docker installation.