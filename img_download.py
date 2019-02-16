from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

paths = response.download({
    "keywords":"food",
    "limit":100,
    "print_urls":True
})
