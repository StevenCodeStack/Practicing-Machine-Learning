import requests
from pathlib import Path
if Path("helper_functions.py").is_file():
    print("Helper Function already exist")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py","wb") as f:
        f.write(request.content)