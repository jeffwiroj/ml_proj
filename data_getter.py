
import gzip
import pandas as pd
import requests
from PIL import Image
import io
from os import listdir
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer


def get_images():
    df = pd.read_pickle("final_frame.pkl")
    asins, ims = df.index.values, df.imUrl.values
    target_size = (224, 224)

    img_src = "images/"

    tot = len(asins)

    for i in range(tot):
        url = ims[i]
        asin = asins[i]

        # save image
        response = requests.get(url)
        if (response.status_code != 404):
            image = Image.open(io.BytesIO(response.content))
            image = image.resize(target_size)
            image.save(img_src + asin + ".jpg")

        print(str(i) + " / " + str(tot), end='\r')

    print()
    print("Download completed.")


if __name__ == "__main__":
    get_images()
