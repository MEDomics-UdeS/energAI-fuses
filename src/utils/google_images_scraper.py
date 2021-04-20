from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

amps = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 110, 125, 150, 175, 200, 225, 250, 300, 350, 400,
        450, 500, 600]

for i in amps:
    keywords = "English Electricr C" + str(i) + "J"

    arguments = {
        "keywords": keywords,
        "limit": 100,
        "save_source": "source_" + keywords,
        "chromedriver": 'C:/Users/simon.giard-leroux/Google Drive/Maîtrise/Python/fuseFinder/chromedriver.exe'
        #                "extract_metadata":True
    }
    paths = response.download(arguments)
