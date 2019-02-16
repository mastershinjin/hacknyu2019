# Food Poisoning
Tries to detect if a picture of produce is fresh or rotten.

Made for HackNYU 2019

# Setup
  - Run: `pip install -r "requirements.txt"`
  - Create folders named `models` and `data` in main folder
  - Create folders named `train` and `test` in the `data` folder
  - Create folders in `train` and `test` named `p` and `n` for positive/negative images
  - Run: `img_setup.py` to generate csv's
  
# Usage
  - Train: `food_poisoning.py train`
  - Test: `food_poisoning.py test [model path]`
  - Run: `food_poisoning.py run [image path] [model path]`