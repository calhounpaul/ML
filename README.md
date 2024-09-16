# ML Repository

This repository is a comprehensive toolkit for advanced machine learning and AI tasks, focusing on natural language processing, image processing, model fine-tuning, and various AI-powered applications. It provides a well-organized structure of scripts, tools, and configurations to support various aspects of machine learning workflows.

## Key Features

- **Environment Setup**: Scripts for setting up CUDA, Docker, and other prerequisites.
- **Model Implementations**: Support for text generation (GPT-like models), image processing (Stable Diffusion, Segment Anything Model), and more.
- **Fine-tuning Tools**: Scripts and configurations for fine-tuning language models.
- **Containerization**: Extensive use of Docker for reproducible and portable ML environments.
- **Integration**: Works with popular platforms like Hugging Face and frameworks such as PyTorch and Transformers.
- **Full Pipeline Support**: Includes tools for data processing, model training, and inference.
- **Multimodal AI**: Capabilities for processing and generating text, images, and videos.

## Repository Structure

- `permathings/`: Core scripts and libraries
  - `libs/`: Utility libraries
    - `digest_git.py`: Tool for analyzing git repositories
    - `secretary.py`: Handles secret management
    - `docker_tools.py`: Docker utility functions
    - `selenium_tools.py`: Selenium-based web scraping tools
  - `prereqs/`: Setup scripts for CUDA, Docker, and other dependencies
  - `scripts/`: Various ML task-specific scripts
    - `stable_diffusion/`: Scripts for running Stable Diffusion models
    - `textgen_webui/`: Text generation web UI setup
    - `tts_webui/`: Text-to-Speech web UI setup
    - `segmentation/`: Image segmentation tools (SAM, OpenAdapt, Segment-and-Track-Anything)
    - `vllm/`: vLLM server setup and examples
    - `tformers_template/`: Transformer model template
    - `ollama/`: Ollama model runner and examples
    - `cogx_video/`: CogX video generation scripts
    - `LLaVA-NeXT/`: LLaVA model setup
    - `langchain/`: Langchain setup and examples
    - `webscraping/`: Web scraping tools
    - `finetuning/`: Scripts for model fine-tuning
    - `ebook_processing/`: Tools for processing and analyzing e-books
    - `document_processing/`: Document analysis tools
    - `OpenDevin/`: OpenDevin model runner
    - `gemma2_RIG/`: Scripts for running the Gemma2 model with Retrieval-Induced Generation
- `persistent_output_examples/`: Sample outputs and analyses

## Key Components

- **Text Generation**: Includes scripts for running instances of text generation webUI, vLLM, and Ollama.
- **Video Generation**: Uses the CogX-5b model to generate silent color video clips:
  ![Jackalope GIF](permathings/persistent_output_examples/a_peculiar_creature__part_rabb.gif)
  ![Shangri-La GIF](permathings/persistent_output_examples/in_the_hidden_valley_of_shangr.gif)
- **Image Processing**: Tools for Stable Diffusion, Segment Anything Model (SAM), and other image segmentation tasks.
- **LLM Integration**: Scripts for running various Large Language Models, including LLaVA, OpenDevin, Ollama models, and Gemma2 with RIG (Retrieval-Induced Generation).
- **Fine-tuning**: Tools for fine-tuning language models with different techniques, including LoRA and QLoRA.
- **Data Processing**: Scripts for e-book and multimodal document processing, including furniture assembly manual analysis:
  ```json
  {
    "manual_filename": "malm-bed-frame-white__AA-2543855-1-100",
    "p35v_description": "This instruction manual is about assembling a furniture piece, specifically a bed frame, as indicated by the diagrams and illustrations showing the assembly process."
  },
  {
    "manual_filename": "mittzon-underframe-for-sit-stand-desk-electric-black__AA-2445413-2-100",
    "p35v_description": "This instruction manual is about assembling a MITTZON desk."
  }
  ```
    and Pixtral-12b inference:

  ![collection of objects](permathings/scripts/vllm_pixtral/objects.jpg)

    *"The image showcases a diverse assortment of small objects spread over a white surface. Some notable items include a colorful frog toy, a pool ball with the number "2", a toy octopus, several keys, a ladybug figurine, a toy shark, a pair of sunglasses, dice, a wooden clothespin, and various buttons and trinkets. Additionally, there are numbers, letters, and an assortment of other miscellaneous items. The background is plain white, which emphasizes the vibrant colors and varied textures of the objects."*

- **Web Interfaces**: Docker configurations for running web-based interfaces for various AI tasks.
- **Langchain**: A docker environment and basic RAG example:
  
  > *Question: Ignat is caught grinning in front of a mirror by Mavra Kuzminichna, and then he has to make tea for which of his relatives?*
  > *Answer: Ignat has to make tea for his grandfather.*
  
- **JSON Outline Generation**: Utilizes vLLM with Mistral to generate structured JSON outlines for various tasks, as demonstrated in the `vllm_json_outlines` scripts.

## Requirements

- CUDA-compatible GPU
- Ubuntu 24.04 (or compatible Linux distribution)
- Docker
- Python 3.x
- Various Python libraries (requirements are specified in individual scripts)

## Setup

0. This repo is currently designed to be deployed inside a fresh Ubuntu 24.04 VM running on an XCP-ng server. It should also be compatible with an instance from a cloud provider like AWS.

1. Clone the repository:
   ```bash
   git clone https://github.com/calhounpaul/ML.git
   cd ML
   ```

2. Set up the environment (tested on Ubuntu 24.04 VM in XCP-ng, but should work with any cloud provider):
   ```bash
   cd ML/permathings/prereqs
   bash ./all.sh
   ```

3. Initialize secrets:
   ```bash
   cd ../scripts/utils
   bash ./init_secrets.sh
   ```

4. Choose a specific task or model from the `scripts/` directory and follow the instructions in the respective script or README.

Note: Some scripts may require significant GPU memory. Adjust configurations as needed based on your available hardware.

## Usage Examples

1. Generate a JSON outline for a recurrence rule:
   ```bash
   cd permathings/scripts/vllm_json_outlines
   ./run_docker_api_server.sh
   python examples/recurrence_rule.py
   ```

2. Run Gemma2 with RIG:
   ```bash
   cd permathings/scripts/gemma2_RIG
   ./run.sh
   python workspace/test.py
   ```

## Additional Features

- **Secret Management**: Utilizes a custom `secretary.py` script for secure handling of API tokens and other sensitive information.
- **Git Repository Analysis**: Includes a `digest_git.py` tool for analyzing and summarizing git repositories.
- **Web Scraping**: Selenium-based tools for web scraping and data collection.
- **Multimodal AI**: Integration of text, image, and video generation capabilities.
- **Adaptive Fine-tuning**: Scripts for fine-tuning models on custom datasets, including e-book content.
- **Document Analysis**: Tools for processing and analyzing various document types, including e-books and instruction manuals.
