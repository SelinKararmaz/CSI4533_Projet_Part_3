import os
from PIL import Image
from utils import model, tools
from utils.histogramCalculator import *
from utils.tools import *
import torch

# cam0 or cam1
SOURCE_PATH = "images/images/"

def find_people(cam):
    """
    Identifies the 5 persons in images
    Draws bounding box and save the images to output folder
    """
    
    source_path_dir = SOURCE_PATH + cam
    images = os.listdir(source_path_dir)

    for index in range(4,5):
        image = cv.imread("input/person_" + str(index) + ".png")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_half = crop_image_half(image)
        Image.fromarray(image.astype(np.uint8)).show()
        person_full = get_rgb_histogram(image)
        person_half = get_rgb_histogram(image_half)
        
        print("Now identifying person " + str(index))

        #750 - second person
        #25 - first person
        #0 - third person
        #20 - fourth person
        for image_name in images:

            image_path = os.path.join(source_path_dir, image_name)
            image = Image.open(image_path)
     
            masked = apply_saved_mask(image,1000,image_name[:-4], cam)
            result = masked[0]
            result_half = masked[1]
            
            smallest_dif = -2
            closest_person = 0
            
            for i in range(len(result)):
                person_in_image = result[i]
                person_in_image_half = result_half[i]
                
                person_in_image_histogram = get_rgba_histogram(person_in_image)
                person_in_image_histogram_half = get_rgba_histogram(person_in_image_half)
                
                difference_1 = compare_correlation(person_full,person_in_image_histogram)
                difference_2 = compare_correlation(person_half,person_in_image_histogram)
                difference_3 = compare_correlation(person_full,person_in_image_histogram_half)
                difference_4 = compare_correlation(person_half,person_in_image_histogram_half)
            
                local_max = max(difference_1,difference_2,difference_3,difference_4)
                
                if(local_max == difference_1 or local_max == difference_2):
                    if(smallest_dif < local_max):
                        smallest_dif = local_max
                        closest_person = person_in_image
                
                if(local_max == difference_3 or local_max == difference_4):
                    if(smallest_dif < local_max):
                        smallest_dif = local_max
                        closest_person = person_in_image
                        
            print(smallest_dif)
            
            correl_threshold = 0.91
            
            if (i == 4):
                correl_threshold = 0.50
            
            if(smallest_dif < correl_threshold): 
                continue
        
            bounding_box_image = get_bounding_box(closest_person, image)
            
            output_folder_path = "output/" + cam + "_output/person_" + str(index) +"/"+image_name
            
            bounding_box_image.save(output_folder_path)
        
        print("Done identifying person " + str(index))

def save_numpy(cam):
    """
    Runs CNN on all the images to detect persons
    Return a numpy array for each cam
    """
    
    source_path_dir = SOURCE_PATH + cam
    
    images = os.listdir(source_path_dir)

    for image_name in images:
        seg_model, transforms = model.get_model()
    
        image_path = os.path.join(source_path_dir, image_name)
        image = Image.open(image_path)
        transformed_img = transforms(image)

        with torch.no_grad():
            output = seg_model([transformed_img])

        save_masks(output,image,image_name[:-4], cam)
           
           
if __name__ == "__main__":
    
    # save numpy files
    # save_numpy("cam0")
    # save_numpy("cam1")

    # Identify people for cam 0
    print("Processing cam 0")
    find_people("cam0")
    
    # Identify people for cam 1
    print("Processing cam 1")
    find_people("cam1")