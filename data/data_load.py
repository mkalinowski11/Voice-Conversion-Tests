import gdown
import os
import zipfile

URL = "https://drive.google.com/file/d/1oLrqcaBa5cDcwNaEOWL7ZRiVy_cTyTji/view?usp=sharing"
DATA_DIR = os.path.join(os.getcwd(), "data")
DATA_PATH = os.path.join(DATA_DIR, "data.zip")

if __name__ == "__main__":
    print(os.getcwd())
    gdown.download(URL, DATA_PATH, quiet=False,fuzzy=True)
    with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)