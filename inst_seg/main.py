import os
from PIL import Image
from utils import model, tools
from utils.histogramCalculator import *
from utils.tools import *
import torch

# Point d'entrée principal du script
if __name__ == "__main__":

    # Définir les répertoires source et de sortie, et le nom de l'image
    source_path_dir = "../images/images/cam0"
    output_path_dir = "examples/output"
    images = os.listdir(source_path_dir)
    output_text_file_path = output_path_dir + "/output.txt"


    for index in range(1,6):
        image = cv.imread("../images/images/five_people" + "/person_" + str(index) + ".png")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_half = crop_image_half(image)
        
        #Image.fromarray(image.astype(np.uint8)).show()
        
        person_full = get_rgb_histogram(image)
        person_half = get_rgb_histogram(image_half)

        for image_name in images[:]:
            
            # Charger le modèle et appliquer les transformations à l'image
            seg_model, transforms = model.get_model()

            # Ouvrir l'image et appliquer les transformations
            image_path = os.path.join(source_path_dir, image_name)
            image = Image.open(image_path)
            transformed_img = transforms(image)

            # Effectuer l'inférence sur l'image transformée sans calculer les gradients
            with torch.no_grad():
                output = seg_model([transformed_img])

            # Traiter le résultat de l'inférence
            result = process_inference(output,image)[0]
            result.save(os.path.join(output_path_dir, image_name))
            
            masks = process_inference(output,image)[1]
            result_str = str(masks)

            masked = apply_saved_mask(image, 1000)

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

                
                #print(compare_correlation(person_full,person_histogram))
           # Image.fromarray(closest_person.astype(np.uint8)).show()
            bounding_box_image = get_bounding_box(closest_person, image)
            
            output_folder_path = "../output/person_" + str(index) +"/"+image_name
            
            bounding_box_image.save(output_folder_path)
            # result.show()e4
        

