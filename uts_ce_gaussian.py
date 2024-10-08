# %%
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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
    'UCF50/YoYo': 'YoYo',
    'UCF50/PlayingGuitar': 'PlayingGuitar'
}
output_folder = 'output/raw'

for input_folder, category_folder in input_folders.items():
    process_all_videos_in_folder(input_folder, category_folder, output_folder)
    
print("All videos in all folders processed successfully!")
    

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
output_folder = 'output/grayscale'
input_folders = {
    'output/raw/PlayingGuitar' : 'PlayingGuitar',
    'output/raw/YoYo' : 'YoYo'
}

for input_folder, category_folder in input_folders.items():
    process_images_in_folder(input_folder, category_folder, output_folder)


# %%
def contrastStretching(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f'Image is not found')
        return 

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # step 1: convert image ke format float
    img_float = grayscale.astype(np.float32)

    # step 2: mendapatkan value min dan max dari image_float
    min_val = np.min(img_float)
    max_val = np.max(img_float)

    # step 3: menset value batas min dan max
    min_out = 0
    max_out = 255

    # step 4: melakukan stretching image 
    stretched = ((img_float - min_val) / (max_val - min_val)) * (max_out - min_out) + min_out

    # step 5: konversi ke uint8
    stretched_image = np.clip(stretched, 0, 255).astype(np.uint8)
    cv2.imwrite(output_folder, stretched_image)
    return stretched_image

# %%
def ce_images_in_folder(input_folder, category_folder, output_folder):
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
                    contrastStretching(input_image_path, output_image_path)
                    print(f"Grayscale image saved to {output_image_path}")

# %%
output_folder = 'output/ce'
input_folders = {
    'output/raw/PlayingGuitar' : 'PlayingGuitar',
    'output/raw/YoYo' : 'YoYo'
}

for input_folder, category_folder in input_folders.items():
    ce_images_in_folder(input_folder, category_folder, output_folder)

# %%
def gaussianBlur(image_path, output_folder, kernel_size):
    # Ensure the kernel size is an odd integer
    if kernel_size % 2 == 0:
        print("Error: Kernel size must be an odd integer.")
        return
    
    # Read the image from the given path
    image = cv2.imread(image_path)
    
    # Check if the image is loaded properly
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Apply Gaussian blur with the specified kernel size
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Save the blurred image to the output folder
    cv2.imwrite(output_folder, blurred_image)
    
    print(f"Blurred image saved to {output_folder}") 

# %%
def gaussian_blur_images_in_folder(input_folder, category_folder, output_folder):
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
                    gaussianBlur(input_image_path, output_image_path, 3)
                    print(f"Filtered image saved to {output_image_path}")

# %%
output_folder = 'output/gaussian'
input_folders = {
    'output/ce/PlayingGuitar' : 'PlayingGuitar',
    'output/ce/YoYo' : 'YoYo'
}

for input_folder, category_folder in input_folders.items():
    gaussian_blur_images_in_folder(input_folder, category_folder, output_folder)



# %%
def calculateMSE(imageA, imageB):
    # Ensure the images have the same dimensions
    if imageA.shape != imageB.shape:
        raise ValueError("Error: Images must have the same dimensions.")
    
    # Compute the MSE
    mse = np.mean((imageA - imageB) ** 2)
    return mse 

# %%
def calculateMeanMSE(input_folder, output_folder):
    input_images = sorted(os.listdir(input_folder))
    output_images = sorted(os.listdir(output_folder))
    
    # Ensure both folders contain the same number of images
    if len(input_images) != len(output_images):
        print("Error: The number of images in both folders must be the same.")
        return
    
    total_mse = 0
    image_count = 0
    mse_array = []
    for input_image_name, output_image_name in zip(input_images, output_images):
        input_image_path = os.path.join(input_folder, input_image_name)
        output_image_path = os.path.join(output_folder, output_image_name)
        
        # Read the input and output images
        input_image = cv2.imread(input_image_path)
        output_image = cv2.imread(output_image_path)
        
        if input_image is None or output_image is None:
            print(f"Error: Unable to load image pair {input_image_name} and {output_image_name}.")
            continue
        
        try:
            # Calculate MSE for the current image pair
            print(f"try calculate MSE for {input_image_name}")
            mse = calculateMSE(input_image, output_image)
            mse_array.append(mse)
            total_mse += mse
            image_count += 1
        except ValueError as e:
            print(e)
            continue
    
    if image_count == 0:
        print("No valid image pairs found.")
        return
    
    # Calculate the average MSE
    average_mse = total_mse / image_count
    print(f"Average MSE: {average_mse}")

    return mse_array, average_mse

# %%
input_folder = 'output/grayscale/PlayingGuitar/v_PlayingGuitar_g01_c01'
output_folder= 'output/gaussian/PlayingGuitar/v_PlayingGuitar_g01_c01'

array_mse, average_mse = calculateMeanMSE(input_folder, output_folder)
print(average_mse)
print(array_mse)



# %%



