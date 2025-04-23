# Brain Tumor Segmentation in Multi-Modal MRI Using Deep Learning

This repository contains the code for the CAP5610 - Machine Learning final project on brain tumor segmentation using Standard U-Net and Attention U-Net models on the BraTS 2021 dataset.

## Dataset

The dataset used in this project is the BraTS 2021 dataset, available on Kaggle: [BraTS 2021 Task 1 Dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1).

1. Download the dataset from the link above.
2. Extract the dataset and place it in a directory named `Dataset/BraTS2021_Training_Data`.

## Requirements

Ensure you have the following prerequisites installed:

- Python 3.11.4
- CUDA 11.8 (if using GPU)
- GCC 12.2.0

### Setup Instructions

1. **Clone the Repository**\
   Clone this repository to your local machine:

   ```
   git clone https://github.com/anirudhnbharadwaj/BraTS-2021.git
   cd BraTS-2021
   ```

2. **Create a Virtual Environment**\
   Set up a virtual environment and activate it:

   ```
   python -m venv brain_seg_env
   source brain_seg_env/bin/activate  # On Windows: brain_seg_env\Scripts\activate
   ```

3. **Install Dependencies**\
   Install the required packages using the `requirements.txt` file:

   ```
   pip install -r requirements.txt
   ```

4. **Set Up Weights & Biases (W&B)**\
   Export your W&B API key for logging (replace `<your-api-key>` with your actual key):

   ```
   export WANDB_API_KEY=<your-api-key>
   ```

## Running the Code

### Option 1: Run on a Local Machine

1. Ensure the dataset is placed in `Dataset/BraTS2021_Training_Data`.

2. Run the main script with the desired model (`StdUNet`, `AttUNet`, or `both`):

   ```
   python main.py --config config/hyperparams.json --model AttUNet
   ```

   - Use `--model StdUNet` for Standard U-Net, `--model AttUNet` for Attention U-Net, or `--model both` to train both models.
   - Hyperparameters can be modified in `config/hyperparams.json` (e.g., batch size, learning rate, epochs).

3. Check the outputs in the `outputs_StdUNet` or `outputs_AttUNet` directories, including logs, checkpoints, and visualizations.

### Option 2: Run on a Cluster (Using SLURM)

1. Ensure the dataset is placed in `Dataset/BraTS2021_Training_Data` on the cluster.

2. Modify the `run_braTS_WB.slurm` script if needed (e.g., update paths, email, or resources).

3. Submit the SLURM job:

   ```
   sbatch run_braTS_WB.slurm
   ```

   - The script is set to run the Attention U-Net model (`AttUNet`). To change the model, edit the `python main.py` command in `run_braTS_WB.slurm` to use `--model StdUNet` or `--model both`.
   - Hyperparameters can be modified in `config/hyperparams.json` (e.g., batch size, learning rate, epochs).

4. Check the SLURM logs in the `slurm_logs` directory and outputs in `outputs_AttUNet`.

## Outputs

- **Training Logs**: Stored in `outputs_<model>/training.log`.
- **Visualizations**: Metrics plots (e.g., `dice.png`, `loss.png`) and prediction images in `outputs_<model>`.
- **Checkpoints**: Model checkpoints in `outputs_<model>/checkpoints`.
- **W&B Logs**: If online, logs are synced to Weights & Biases; if offline, check `outputs_<model>/wandb` for logs to sync later.
