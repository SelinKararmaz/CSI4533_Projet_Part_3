import cv2 as cv

# Number of bins for each channel
b_bins = 100
g_bins = 100
r_bins = 100

# Range for each channel
b_range = [0, 256]
g_range = [0, 256]
r_range = [0, 256]
ranges = b_range + g_range + r_range  # Concatenate lists

def get_rgb_histogram(image):
    # Assuming b_bins, g_bins, r_bins, and ranges are defined elsewhere in your code
    ranges = [0, 256, 0, 256, 0, 256]

    # Split the image into its r g b channels
    b, g, r = cv.split(image)   
    # Combine the histograms for each channel using the predefined bins and ranges
    hist_rgb = cv.calcHist([image], [0, 1, 2], None, [b_bins, g_bins, r_bins], ranges, accumulate=False)
    # Normalizes the histogram
    cv.normalize(hist_rgb, hist_rgb, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return hist_rgb
def get_rgba_histogram(image):
    # Assuming b_bins, g_bins, r_bins, and ranges are defined elsewhere in your code
    ranges = [0, 256, 0, 256, 0, 256]
    
    # Split the image into its RGB channels
    b, g, r, a = cv.split(image)

    # Create a mask for opaque regions (where alpha is not 0)
    mask = (a > 0)

    # Apply the mask to the RGB channels
    b = b[mask]
    g = g[mask]
    r = r[mask]

    # Combine the histograms for each channel using the predefined bins and ranges
    hist_rgb = cv.calcHist([b, g, r], [0, 1, 2], None, [b_bins, g_bins, r_bins], ranges, accumulate=False)
    
    # Normalize the histogram
    cv.normalize(hist_rgb, hist_rgb, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    
    return hist_rgb

# Compares 2 images
# Returns 1 if images are same
def compare_correlation(hist_base, hist_compare): 
    # 0 = Correlation comparator   
    comparaison = cv.compareHist(hist_base, hist_compare, 0)
    return comparaison


