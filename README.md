# Scene Grounding in Dense Visual Environments

This project is a solution for the "Scene Localization in Dense Images via Natural Language Queries" task. The system identifies and localizes a specific region within a dense image that corresponds to a given natural language description.

This implementation serves as a robust baseline, developed as part of the AIMS 2K28 Recruitments problem statement. The submission deadline for this project is **August 15, 2025, 11:59 pm**.

## Table of Contents
- [Project Objective](#project-objective)
- [Methodology](#methodology)
- [Demonstration](#demonstration)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Development Challenges](#development-challenges)

## Project Objective

The goal is to build a model that can parse a complex visual scene and ground a textual query within it. Given a high-resolution image with multiple activities (e.g., a busy market) and a query like "a vendor selling vegetables to a customer," the model must output the specific bounding box that contains this described event.

- **Input**: A single dense image and a free-form text query.
- **Output**: The bounding box coordinates of the corresponding region in the image.

## Methodology

This system uses a two-stage "propose-then-rank" architecture, which was chosen for its stability and effectiveness.

1.  **Region Proposal**: The **Selective Search** algorithm is used to generate a set of potential object regions (candidate bounding boxes) from the input image.
2.  **Ranking with CLIP**: The pre-trained multimodal **CLIP model (ViT-B/32)** from OpenAI is used to find the best match.
    - A vector embedding is generated for the text query.
    - A vector embedding is generated for each proposed image region.
    - **Cosine similarity** is calculated between the text embedding and each image region's embedding.
    - The bounding box of the region with the highest similarity score is returned as the final result.

## Demonstration

A short video demonstrating the system in action with multiple queries can be found below.

- **[Link to Demo Video Here]**

## Setup and Installation

To run this project locally, you need to have Python 3 and the following libraries installed.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/scene-grounding-project.git](https://github.com/your-username/scene-grounding-project.git)
    cd scene-grounding-project
    ```

2.  Install the required packages:
    ```bash
    pip install torch torchvision ftfy regex tqdm opencv-contrib-python matplotlib
    pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
    ```

## How to Run

This notebook was developed and is best run in a **Kaggle environment**, as it ensures all dependencies and environment configurations are handled correctly.

**To run on Kaggle (Recommended):**
1.  Upload the `scene-grounding-project.ipynb` notebook to your Kaggle account.
2.  Open the notebook and simply run all the cells in order. The required packages are installed within the notebook itself.

**To run locally:**
1.  Ensure all dependencies from the [Setup and Installation](#setup-and-installation) section are installed.
2.  Open the `scene-grounding-project.ipynb` notebook in a Jupyter environment.
3.  Execute the cells in order from top to bottom.
4.  You can modify the `IMAGE_URL` and `TEXT_QUERY` variables in the "Demonstration" section to test with your own images and descriptions.

The final cell will display the input image with the predicted bounding box drawn on it.

## Development Challenges

During development, several alternative approaches were explored but ultimately abandoned due to technical constraints.

- **End-to-End Models (MDETR)**: An attempt to use a more advanced, single-stage model like MDETR failed due to persistent and unresolvable library import errors in the Kaggle environment.
- **Dependency Issues**: Significant time was spent resolving environment-specific issues, including NumPy ABI incompatibilities, CUDA-hardware mismatches, and missing OpenCV modules.

These challenges led to the selection of the current architecture, which proved to be far more stable and reliable, especially within the Kaggle ecosystem.
