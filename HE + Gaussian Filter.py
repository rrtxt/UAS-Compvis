# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

##MEMPROSES VIDEO UNTUK MENJADI FRAME
# %%
def process_video(input_video, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_path = input_video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{output_folder}/frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    cap.release()

# %%
def process_all_videos_in_folder(input_folder, category_folder, output_folder):
    category_output_folder = os.path.join(output_folder, category_folder)
    
    for video_filename in os.listdir(input_folder):
        if video_filename.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(input_folder, video_filename)
            
            video_name = os.path.splitext(video_filename)[0]
            output_folder = os.path.join(category_output_folder, video_name)
            
            print(f"Processing {video_filename}")
            process_video(video_path, output_folder)
            print(f"Frames saved to {output_folder}")

# %%
input_folders = {
    'YoYo': 'YoYo',
    'PlayingGuitar': 'PlayingGuitar'
}
output_folder = 'Output/raw'

for input_folder, category_folder in input_folders.items():
    process_all_videos_in_folder(input_folder, category_folder, output_folder)
    
print("All videos in all folders processed successfully!")
    

##PROSES GRAYSCALE
# %%
def grayscale_image(input_image_path, output_image_path):
    # Read the image in color
    image = cv2.imread(input_image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image
    cv2.imwrite(output_image_path, gray_image)

# %%
def process_images_in_folder(input_folder, category_folder, output_folder):
    # Iterate through all subfolders in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Create a mirrored subfolder in the output folder
            output_subfolder_path = os.path.join(output_folder, category_folder, subfolder)
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            # Process all images in the subfolder
            for image_filename in os.listdir(subfolder_path):
                if image_filename.endswith(('.png', '.jpg', '.jpeg')):
                    input_image_path = os.path.join(subfolder_path, image_filename)
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    
                    print(f"Processing image {image_filename} in {subfolder}...")
                    grayscale_image(input_image_path, output_image_path)
                    print(f"Grayscale image saved to {output_image_path}")

# %%
output_folder = 'Output/Grayscale'
input_folders = {
    'Output/Raw/PlayingGuitar' : 'PlayingGuitar',
    'Output/Raw/YoYo' : 'YoYo'
}

for input_folder, category_folder in input_folders.items():
    process_images_in_folder(input_folder, category_folder, output_folder)


##PROSES HISTOGRAM EQUALIZED
# %%
def histogram_equalized_image(input_image_path, output_image_path, histogram_path):
    # Read the image in color
    image = cv2.imread(input_image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Save the equalized image
    cv2.imwrite(output_image_path, equalized_image)
    
    # Create a plot for the equalized image and its histogram
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display equalized image
    axs[0].imshow(equalized_image, cmap='gray')
    axs[0].set_title('Histogram Equalized')
    axs[0].axis('off')

    # Calculate and plot the histogram of the equalized image
    axs[1].hist(equalized_image.ravel(), bins=256, range=(0, 256), color='gray')
    axs[1].set_title('Histogram of Equalized Image')

    # Save the histogram plot
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()

# %%
def process_images_in_folder2(input_folder, category_folder, output_folder_equalized, output_folder_histogram):
    # Iterate through all subfolders in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Create mirrored subfolders in both output folders (Equalized_Only and Histogram_Equalized)
            equalized_subfolder = os.path.join(output_folder_equalized, category_folder, subfolder)
            histogram_subfolder = os.path.join(output_folder_histogram, category_folder, subfolder)

            # Ensure the folders exist
            if not os.path.exists(equalized_subfolder):
                os.makedirs(equalized_subfolder)
            if not os.path.exists(histogram_subfolder):
                os.makedirs(histogram_subfolder)

            # Process all images in the subfolder
            for image_filename in os.listdir(subfolder_path):
                if image_filename.endswith(('.png', '.jpg', '.jpeg')):
                    input_image_path = os.path.join(subfolder_path, image_filename)
                    
                    # Paths for saving the equalized images and histograms
                    output_image_equalized_path = os.path.join(equalized_subfolder, image_filename)
                    output_image_histogram_path = os.path.join(histogram_subfolder, image_filename)
                    histogram_output_path = os.path.join(histogram_subfolder, f"histogram_{image_filename}.png")

                    print(f"Processing image {image_filename} in {subfolder}...")

                    # Save equalized image in Equalized_Only folder
                    histogram_equalized_image(input_image_path, output_image_equalized_path, histogram_output_path)
                    print(f"Equalized image saved to {output_image_equalized_path}")
                    
                    # Save equalized image and histogram in Histogram_Equalized folder
                    print(f"Equalized image and histogram saved to {output_image_histogram_path} and {histogram_output_path}")

# %%
# Define the output folders
output_folder_equalized = 'Output/Histogram_Equalized/Tanpa_Histogram'
output_folder_histogram = 'Output/Histogram_Equalized/Dengan_Histogram'

# Define the input folders
input_folders = {
    'Output/Grayscale/PlayingGuitar': 'PlayingGuitar',
    'Output/Grayscale/YoYo': 'YoYo',
}

# Process each input folder
for input_folder, category_folder in input_folders.items():
    print(f"Processing category: {category_folder}")
    process_images_in_folder2(input_folder, category_folder, output_folder_equalized, output_folder_histogram)
    print(f"Finished processing {category_folder}")

##PROSES GAUSSIAN FILTERING
# %% 
# Function to apply Gaussian filter
def gaussian_filter_image(input_image_path, output_image_path, kernel_size=(3, 3), sigma=1.0):
    # Read the image in color
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not read image {input_image_path}")
        return
    
    # Apply Gaussian filtering
    gaussian_image = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Save the Gaussian filtered image
    cv2.imwrite(output_image_path, gaussian_image)
    print(f"Gaussian filtered image saved to {output_image_path}")

# %% 
# Function to process images in a folder
def process_images_with_gaussian(input_folder, category_folder, output_folder, kernel_size=(3, 3), sigma=1.0):
    # Iterate through all subfolders in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Create a mirrored subfolder in the output folder
            output_subfolder_path = os.path.join(output_folder, category_folder, subfolder)
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            # Process all images in the subfolder
            for image_filename in os.listdir(subfolder_path):
                if image_filename.endswith(('.png', '.jpg', '.jpeg')):
                    input_image_path = os.path.join(subfolder_path, image_filename)
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    
                    print(f"Processing image {image_filename} in {subfolder}...")
                    gaussian_filter_image(input_image_path, output_image_path, kernel_size, sigma)
                    print(f"Gaussian filtered image saved to {output_image_path}")

# %% 
# Define input and output folders
output_folder = 'Output/HE+GaussianFilter/3x3'
input_folders = {
    'Output/Histogram_Equalized/Tanpa_Histogram/PlayingGuitar': 'PlayingGuitar',
    'Output/Histogram_Equalized/Tanpa_Histogram/YoYo': 'YoYo'
}

# Process images with Gaussian filtering
for input_folder, category_folder in input_folders.items():
    process_images_with_gaussian(input_folder, category_folder, output_folder, kernel_size=(3, 3), sigma=1.0)
    print(f"Finished processing {category_folder} with Gaussian filtering")

##EVALUASI DENGAN MSE DAN PSNR 
# %%
# Function to calculate MSE
def calculate_mse(original_image, processed_image):
    # Calculate the Mean Squared Error between the original and processed images
    mse = np.mean((original_image - processed_image) ** 2)
    return mse

# Function to calculate PSNR
def calculate_psnr(original_image, processed_image):
    mse = calculate_mse(original_image, processed_image)
    if mse == 0:
        return float('inf')  # Return infinity if no difference
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# %% 
# Function to process subfolders and calculate average MSE and PSNR per subfolder,
# Then calculate average per category and overall average
def process_subfolders_and_calculate_average_per_category(input_folder_original, input_folder_processed, input_folders, output_file):
    total_mse_by_category = {
        'PlayingGuitar': 0.0,
        'YoYo': 0.0
    }
    total_psnr_by_category = {
        'PlayingGuitar': 0.0,
        'YoYo': 0.0
    }
    subfolder_count_by_category = {
        'PlayingGuitar': 0,
        'YoYo': 0
    }

    # Open the output file in write mode (creates a new file for overall results)
    with open(output_file, 'w') as f:
        f.write("--- Overall MSE and PSNR Calculation ---\n")

        # Iterate through all categories (PlayingGuitar and YoYo)
        for category_folder in input_folders.values():
            category_original_path = os.path.join(input_folder_original, category_folder)
            category_processed_path = os.path.join(input_folder_processed, category_folder)

            # Iterate through all subfolders within each category
            for subfolder in os.listdir(category_original_path):
                total_mse = 0.0
                total_psnr = 0.0
                image_count = 0
                
                subfolder_original_path = os.path.join(category_original_path, subfolder)
                subfolder_processed_path = os.path.join(category_processed_path, subfolder)

                # Check if both original and processed subfolders exist
                if os.path.isdir(subfolder_original_path) and os.path.isdir(subfolder_processed_path):
                    # Process all images in the subfolder
                    for image_filename in os.listdir(subfolder_original_path):
                        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
                            original_image_path = os.path.join(subfolder_original_path, image_filename)
                            processed_image_path = os.path.join(subfolder_processed_path, image_filename)

                            # Read both original and processed images
                            original_image = cv2.imread(original_image_path)
                            processed_image = cv2.imread(processed_image_path)

                            if original_image is None or processed_image is None:
                                print(f"Error: Could not read one of the images: {image_filename}")
                                continue  # Skip if image could not be read

                            # Calculate MSE and PSNR
                            mse = calculate_mse(original_image, processed_image)
                            psnr = calculate_psnr(original_image, processed_image)

                            total_mse += mse
                            total_psnr += psnr
                            image_count += 1

                    # Calculate average MSE and PSNR for the current subfolder
                    if image_count > 0:
                        average_mse = total_mse / image_count
                        average_psnr = total_psnr / image_count
                    else:
                        average_mse = 0
                        average_psnr = 0

                    # Write average results for the current subfolder to the file
                    f.write(f"\nSubfolder: {subfolder} (Category: {category_folder})\n")
                    f.write(f"Average MSE: {average_mse}\n")
                    f.write(f"Average PSNR: {average_psnr} dB\n")

                    # Accumulate category-level MSE and PSNR for overall calculation
                    total_mse_by_category[category_folder] += average_mse
                    total_psnr_by_category[category_folder] += average_psnr
                    subfolder_count_by_category[category_folder] += 1

        # Calculate average MSE and PSNR for each category
        f.write("\n--- Average Results by Category ---\n")
        for category in total_mse_by_category.keys():
            if subfolder_count_by_category[category] > 0:
                average_mse_by_category = total_mse_by_category[category] / subfolder_count_by_category[category]
                average_psnr_by_category = total_psnr_by_category[category] / subfolder_count_by_category[category]
            else:
                average_mse_by_category = 0
                average_psnr_by_category = 0

            f.write(f"{category} - Average MSE: {average_mse_by_category}\n")
            f.write(f"{category} - Average PSNR: {average_psnr_by_category} dB\n")

        # Calculate overall average MSE and PSNR across all categories
        overall_mse = (total_mse_by_category['PlayingGuitar'] + total_mse_by_category['YoYo']) / (
            subfolder_count_by_category['PlayingGuitar'] + subfolder_count_by_category['YoYo'] or 1)
        overall_psnr = (total_psnr_by_category['PlayingGuitar'] + total_psnr_by_category['YoYo']) / (
            subfolder_count_by_category['PlayingGuitar'] + subfolder_count_by_category['YoYo'] or 1)

        # Write overall average results to the file
        f.write("\n--- Overall Results ---\n")
        f.write(f"Overall Average MSE: {overall_mse}\n")
        f.write(f"Overall Average PSNR: {overall_psnr} dB\n")

# %%
# Define input folders for original and processed images
input_folder_original = 'Output/Grayscale'
input_folder_processed = 'Output/HE+GaussianFilter/3x3'

input_folders = {
    'PlayingGuitar': 'PlayingGuitar',
    'YoYo': 'YoYo'
}

# Process subfolders, calculate average MSE and PSNR per subfolder, and then calculate average per category
output_file = 'Output/average_by_category_mse_psnr_results.txt'
process_subfolders_and_calculate_average_per_category(input_folder_original, input_folder_processed, input_folders, output_file)

print(f"Finished processing. Average results saved to {output_file}")

# %%
