import numpy as np
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image
import cv2 as cv

def apply_saved_mask(image, threshold, image_name, cam):
    """
    Apply the saved numpy mask to the images
    The output will be cropped images of be the persons detected in the image
    """
   
    # Load mask from numpy file and apply
    masks = np.load(cam + '_npy/'+image_name+'.npy')
    img_np = np.array(image)
    people = []
    people_half = []
    
    # Go through all the masks
    for i, mask in enumerate(masks):  
        
        # Only keep masks above the threshold
        if(np.count_nonzero(mask)< threshold): continue
       
       # create an empty array to store the result
        masked_region = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)

        # Apply the mask to the original image to retain the red mask colored region
        # this creates an image of a person only, with the background being black
        for c in range(3):
            masked_region[:, :, c] = np.where(mask, img_np[:, :, c], 0)

        # Set the alpha channel of the resulting mask to be transparent where the mask is 0 (multiplies mask by image)
        # this creates image of the person, with the background being transparent
        masked_region[:, :, 3] = (mask * 255).astype(np.uint8)
        
        # Calculate bounding box of mask
        non_zero_indices = np.argwhere(mask)
        min_row, min_col = np.min(non_zero_indices, axis=0)
        max_row, max_col = np.max(non_zero_indices, axis=0)
        
        # Calculate the midpoint of the bounding box
        mid_row = (min_row + max_row) // 2
        
        # Get the top half of the masked region
        cropped_mask = masked_region[min_row:mid_row, :, :]

        # Append the masked image to the list
        people.append(masked_region)
        people_half.append(cropped_mask)
      
    return people,people_half


def crop_image_half(image):
    """
    Crop the image in half using the height
    """
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the midpoint of the height
    midpoint = height // 2

    # Extract the top half of the image
    return image[0:midpoint, :]


def get_bounding_box(mask, image):
    """
    Draws the bounding box of the mask on the original image
    """
    
    image = np.asarray(image)
    
    # Extract the alpha channel from the mask
    alpha_channel = mask[:, :, 3]

    # Find non-zero indices in the alpha channel
    non_zero_indices = np.argwhere(alpha_channel)

    # Calculate bounding box coordinates
    min_row, min_col = np.min(non_zero_indices, axis=0)
    max_row, max_col = np.max(non_zero_indices, axis=0)

    # Draw the bounding box on the image
    bounding_box_image = cv.rectangle(image.copy(), (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)

    return Image.fromarray(bounding_box_image)
    

def save_masks(model_output, image,image_name, cam):
    """
    Save the numpy masks that was extracted from the images using CNN to a numpy file
    """
    np_masks = []
    # Extraire les masques, les scores, et les labels de la sortie du modèle
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']

    # Convertir l'image en tableau numpy
    img_np = np.array(image)

    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
    
        # Appliquer le seuil et vérifier si le label correspond à une personne
        if score > THRESHOLD and label == PERSON_LABEL:
            
            # Convertir le masque en tableau numpy et l'appliquer à l'image            
            mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255) 
            np_masks.append(mask_np)            

            for c in range(3):
                img_np[:, :, c] = np.where(mask_np, 
                                        (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                        img_np[:, :, c])
   
    # save mask to text file
    #result = apply_saved_mask(image,1000,np_masks)
    with open(cam + '_npy/'+ image_name +'.npy', 'wb') as f:
        np.save(f,np_masks)