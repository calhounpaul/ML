# ML Repository

This repository is a comprehensive toolkit for advanced machine learning and AI tasks, focusing on natural language processing, image processing, model fine-tuning, and various AI-powered applications. It provides a well-organized structure of scripts, tools, and configurations to support various aspects of machine learning workflows.

## Key Features

- **Environment Setup**: Scripts for setting up CUDA, Docker, and other prerequisites.
- **Model Implementations**: Support for text generation (GPT-like models), image processing (Stable Diffusion, Segment Anything Model), and more.
- **Fine-tuning Tools**: Scripts and configurations for fine-tuning language models.
- **Containerization**: Extensive use of Docker for reproducible and portable ML environments.
- **Integration**: Works with popular platforms like Hugging Face and frameworks such as PyTorch and Transformers.
- **Full Pipeline Support**: Includes tools for data processing, model training, and inference.

## Repository Structure

- `permathings/`: Core scripts and libraries
  - `libs/`: Utility libraries (e.g., `digest_git.py`, `secretary.py`)
  - `prereqs/`: Setup scripts for CUDA, Docker, and other dependencies
  - `scripts/`: Various ML task-specific scripts
    - `stable_diffusion/`: Scripts for running Stable Diffusion models
    - `textgen_webui/`: Text generation web UI setup
    - `tts_webui/`: Text-to-Speech web UI setup
    - `segmentation/`: Image segmentation tools (SAM, OpenAdapt)
    - `vllm/`: vLLM server setup and examples
    - `finetuning/`: Scripts for model fine-tuning
    - `ebook_processing/`: Tools for processing and analyzing e-books
- `persistent_output_examples/`: Sample outputs and analyses

## Key Components

- **Text Generation**: Includes scripts for running instances of text generation webUI and vLLM.
- **Video Generation**: Uses the CogX-5b model to generate silent color video clips:
# ![Jackalope GIF](permathings/persistent_output_examples/a_peculiar_creature__part_rabb.gif)
- **Image Processing**: Tools for Stable Diffusion, Segment Anything Model (SAM), and other image segmentation tasks.
- **LLM Integration**: Scripts for running various Large Language Models, including LLaVA and OpenDevin.
- **Fine-tuning**: Tools for fine-tuning language models with different techniques.
- **Data Processing**: Scripts for e-book and multimodal document processing, including furniture assembly manual analysis:
```
"malm-bed-frame-white__AA-2543855-1-100"
"This instruction manual is about assembling a furniture piece, specifically a bed frame, as indicated by the diagrams and illustrations showing the assembly process."

"mittzon-underframe-for-sit-stand-desk-electric-black__AA-2445413-2-100"
"This instruction manual is about assembling a MITTZON desk."

"uppland-sofa-frame__AA-2202687-1-2"
"This instruction manual is about assembling an IKEA sofa."

"kura-reversible-bed__AA-843436-6_pub"
"This instruction manual is about assembling a piece of furniture, specifically a bed frame."

"neiden-bed-frame__AA-2044866-4_pub"
"p35v_description": "This instruction manual is about assembling a bed frame, specifically a 'NEIDEN' bed frame by IKEA."

"askvoll-bed-frame__AA-942096-6_pub"
"This instruction manual is about assembling a furniture piece, specifically a bed frame, as indicated by the title 'ASKVOLL' and the illustrations showing the assembly process."
```
- **Web Interfaces**: Docker configurations for running web-based interfaces for various AI tasks.

## Requirements

- CUDA-compatible GPU
- Ubuntu 24.04 (or compatible Linux distribution)
- Docker
- Python 3.x
- Various Python libraries (requirements are specified in individual scripts)

## Setup

0. This repo is currently designed to be deployed inside a fresh Ubuntu 24.04 VM running on an XCP-ng server. It should also be compatible with an instance from a cloud provider like AWS.

1. Clone the repository:
   ```
   git clone https://github.com/[username]/ML.git
   cd ML
   ```

2. Set up the environment (tested on Ubuntu 24.04 VM in XCP-ng, but should work with any cloud provider):
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