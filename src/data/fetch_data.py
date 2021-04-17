import requests
import os
import zipfile
from tqdm import tqdm
import sys


def download_file_from_google_drive(file_id, dest, chunk_size=32768):
    """

    Inspired from :
    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

    :param file_id:
    :param dest:
    :param chunk_size:
    :return:
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    if response.ok:
        with open(dest, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size), desc='Please wait...'):
                if chunk:
                    f.write(chunk)
    else:
        print('Error', response.status_code, response.reason)
        sys.exit(1)


if __name__ == "__main__":
    images_id = '12mZB0Or1FhzwZO_zUTYCbbuxwQ56-5PP'
    annotations_id = '1y3BbSGF98Cs9eOTJF-V32JJ8s9dodSC-'

    for i in range(2):
        os.chdir(os.path.pardir)

    images_unzip = os.path.join(os.getcwd(), 'data', 'raw')
    images_dest = os.path.join(images_unzip, 'images.zip')
    annotations_dest = os.path.join(os.getcwd(), 'data', 'annotations', 'annotations.csv')

    print('Downloading images to:\t', images_unzip)
    download_file_from_google_drive(images_id, images_dest)
    print('Done!\nUnzipping images...')

    with zipfile.ZipFile(images_dest, 'r') as zip_ref:
        zip_ref.extractall(images_unzip)

    os.remove(images_dest)

    print('Done!\n\nDownloading annotations to:\t', annotations_dest)
    download_file_from_google_drive(annotations_id, annotations_dest)
    print('Done!')
