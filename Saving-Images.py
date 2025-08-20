from PIL import Image

def save_images(images, filenames):
    """Save generated images to files"""
    for image, filename in zip(images, filenames):
        image.save(filename)
