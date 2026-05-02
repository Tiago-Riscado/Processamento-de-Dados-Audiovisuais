import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR      = os.getenv("DATA_DIR",      "./data/dataset_waste_container")
BALANCED_DIR  = os.getenv("BALANCED_DIR",  "./data/dataset_augmented")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",    "./data/dataset_split")
RESULTS_DIR   = os.getenv("RESULTS_DIR",  "./results")

MODEL_RESNET   = os.getenv("MODEL_RESNET",   "./models/best_resnet18_model.pth")
MODEL_MOBILENET = os.getenv("MODEL_MOBILENET", "./models/best_model_MobilNet.pth")
OUTPUT_CSV_RESNET    = os.getenv("OUTPUT_CSV_RESNET",    "./results/scores_resnet.csv")
OUTPUT_CSV_MOBILENET = os.getenv("OUTPUT_CSV_MOBILENET", "./results/scores_mobilenet.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Dataset ---
IMG_SIZE               = 224
TARGET_IMAGES_PER_CLASS = 1000
VAL_IMAGES             = 200
TEST_IMAGES            = 200
CLASS_TO_REMOVE        = "container_ash"

# --- Training ---
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LEARNING_RATE = 0.0005
PATIENCE      = 5

# --- ImageNet normalisation ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
