import os
from PIL import Image

def filter_images_in_subdirectories(root_directory, max_dimension=512):
    """
    Deletes images in subdirectories of the given root directory
    if at least one spatial dimension is greater than max_dimension.
    Counts and displays the total number of images left after filtering.
    
    Args:
        root_directory (str): Path to the root directory containing subdirectories.
        max_dimension (int): Maximum allowed dimension for both width and height.
    """
    if not os.path.exists(root_directory):
        print(f"Directory {root_directory} does not exist.")
        return

    total_images_left = 0

    for subdir, _, files in os.walk(root_directory):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)

            # Skip non-image files
            if not file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
                continue

            try:
                # Open the image
                with Image.open(file_path) as img:
                    width, height = img.size

                    # Check if the image has dimensions exceeding the limit
                    if width > max_dimension or height > max_dimension:
                        print(f"Keeping {file_path}: Size {width}x{height}")
                        total_images_left += 1
                    else:
                        print(f"Deleting {file_path}: Size {width}x{height} is within the limit")
                        os.remove(file_path)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"\nTotal number of images left: {total_images_left}")

# Example usage
root_directory_path = "/projects/ydlin/OutdoorSceneTrain_v2/"
filter_images_in_subdirectories(root_directory_path, max_dimension=512)