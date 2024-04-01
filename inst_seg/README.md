
# Instance Segmentation

This project was tested with Python 3.10.

## Instructions

Follow these simple steps to get your local copy up and running.

1. Download the folder `images.zip` from the google drive link below:
   ```
   https://drive.google.com/file/d/1potC4tmKjvLAlXSmhaGH59u-g5u4qg5-/view?usp=sharing
   ```

2. Extract `images.zip` and place it in the root directory of the repository.

3. Open a command line on the root directory.

4. Create a virtual environment
   ```
   python3 -m venv env
   ```

5. Activate the virtual environment
   ```
   source env/bin/activate
   ```

6. Install code dependencies
   ```
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
7. Create a folder `five_people` in the `images\images` folder and place input images of the 5 people in `images\images\five_people` folder

## Utilisation

Run the code with `python main.py`
