import os
import urllib.request
import gzip

urls = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
]

workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
if workspace_dir:
    data_dir = os.path.join(workspace_dir, "data")
else:
    data_dir = "data"

os.makedirs(data_dir, exist_ok=True)

for url in urls:
    filename = os.path.join(data_dir, os.path.basename(url))
    uncompressed_filename = filename[:-3]

    if os.path.exists(uncompressed_filename):
        print(f"{uncompressed_filename} already exists, skipping.")
        continue

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)

    print(f"Uncompressing {filename}...")
    with gzip.open(filename, "rb") as f_in:
        with open(uncompressed_filename, "wb") as f_out:
            f_out.write(f_in.read())

    os.remove(filename)
    print(f"Done with {uncompressed_filename}")
