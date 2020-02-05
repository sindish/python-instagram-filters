import cv2
import numpy as np

def histogram_transfer(color_img, content_img):
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB) 
    content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2LAB)
    
    L_color, a_color, b_color = cv2.split(np.float32(color_img))
    L_content, a_content, b_content = cv2.split(np.float32(content_img))
    
    L_content -= L_content.mean()
    L_content *= (L_content.std() / L_color.std()) 
    L_content += L_color.mean()

    a_content -= a_content.mean()
    a_content *= (a_content.std() / a_color.std()) 
    a_content += a_color.mean()

    b_content -= b_content.mean()
    b_content *= (b_content.std() / b_color.std()) 
    b_content += b_color.mean()
    
    output = np.stack([L_content, a_content, b_content], axis=-1)
    output = np.uint8(np.clip(output, 0, 255))
    output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
    
    return output
