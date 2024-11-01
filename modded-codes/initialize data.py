import os
from collections import defaultdict
import cv2

def list_png_files_in_directory(directory):
    """
    Walk through the given directory and list all .png files and subdirectories,
    and count the total number of .png files.

    Args:
        directory (str): The path to the directory to explore.

    Returns:
        None
    """
    total_png_count = 0

    try:
        for root, dirs, files in os.walk(directory):
            print(f"Current Directory: {root}")
            png_files = [file for file in files if file.lower().endswith('.png')]
            if png_files:
                for file_name in png_files:
                    print(f"  PNG File: {file_name}")
                total_png_count += len(png_files)
            else:
                print("  No PNG files found.")
            print()  # Add a newline for better readability

        #print(f"Total number of .png files found: {total_png_count}") #used to check whether it is able to find files 

    except Exception as e:
        print(f"An error occurred: {e}")


def categorize_png_files(directory):
    """
    Categorize .png files in the given directory based on their names starting with "NOISY" or "GT".

    Args:
        directory (str): The path to the directory to explore.

    Returns:
        dict: A dictionary with categories as keys and lists of file paths as values.
    """
    categorized_files = defaultdict(list)

    try:
        for root, dirs, files in os.walk(directory):
            png_files = [file for file in files if file.lower().endswith('.png')]
            for file_name in png_files:
                if file_name.startswith("NOISY"):
                    categorized_files["NOISY"].append(os.path.join(root, file_name))
                elif file_name.startswith("GT"):
                    categorized_files["GT"].append(os.path.join(root, file_name))
                else:
                    categorized_files["Other"].append(os.path.join(root, file_name))
                    
        return categorized_files

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_and_process_images(directory, noisy_output_dir, gt_output_dir):
    """
    Extract .png files starting with "NOISY" and "GT", resize them,
    and save them to specified output directories with renamed files.

    Args:
        directory (str): The path to the directory to explore.
        noisy_output_dir (str): The output directory for "NOISY" images.
        gt_output_dir (str): The output directory for "GT" images.

    Returns:
        None
    """
    # Create output directories if they don't exist
    os.makedirs(noisy_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    try:
        for root, dirs, files in os.walk(directory):
            # Get the name of the current directory
            folder_name = os.path.basename(root)
            for file_name in files:
                if file_name.lower().endswith('.png'):
                    full_path = os.path.join(root, file_name)
                    if file_name.startswith("NOISY"):
                        new_file_name = f"NOISY_{folder_name}.png"
                        save_path = os.path.join(noisy_output_dir, new_file_name)
                        process_and_save_image(full_path, save_path)
                    elif file_name.startswith("GT"):
                        new_file_name = f"GT_{folder_name}.png"
                        save_path = os.path.join(gt_output_dir, new_file_name)
                        process_and_save_image(full_path, save_path)

    except Exception as e:
        print(f"An error occurred: {e}")

def process_and_save_image(input_path, output_path):
    """
    Resize the image to 1024x768 and save it.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.

    Returns:
        None
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(input_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Could not read image: {input_path}")
            return

        # Resize the image to 1024x768
        img_resized = cv2.resize(img, (512, 384))

        # Save the processed image
        cv2.imwrite(output_path, img_resized)
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Call the function with the desired directory
list_png_files_in_directory(r"G:\SIDD_Small_sRGB_Only\SIDD_Small_sRGB_Only\Data")

extract_and_process_images(
    r"G:\SIDD_Small_sRGB_Only\SIDD_Small_sRGB_Only\Data",  # Replace with your input directory
    r"G:\SIDD\NOISY",     # Replace with your NOISY output directory
    r"G:\SIDD\GT"         # Replace with your GT output directory
)
