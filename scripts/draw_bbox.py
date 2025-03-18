import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def is_homogeneous_single(image, threshold=0.06):
    """
    Check if a region is homogeneous by summing the variances of each channel.
    """
    channels = image.shape[0]
    # Calculate the maximum difference for each channel (image shape: channels, height, width)
    channel_differences = torch.max(image.reshape(channels, -1), dim=1)[0] - torch.min(image.reshape(channels, -1), dim=1)[0]  # Variance for each channel
    # Sum the variances across channels to get a single measure of homogeneity
    total_variance = torch.mean(channel_differences)
    
    # Check if the total variance is below the threshold
    return total_variance <= threshold

def quadtree_compress_single(image, x, y, width, height, threshold=0.06, min_size=1):
    """
    Apply quadtree compression to a single image.
    """
    if width <= min_size or height <= min_size:
        # Base case: If the region is small or depth limit is reached, return the bounding box
        bounding_box = [(x, y, x + width, y + height)]
        return bounding_box
    
    # Extract the region of interest from the image
    region = image[:, y:y+height, x:x+width]
    b = is_homogeneous_single(region, threshold)
    
    # Check homogeneity of the region
    if b:
        # If homogeneous, return the bounding box
        return [(x, y, x+width, y+height)]
    
    # Otherwise, subdivide into four quadrants
    half_width = width // 2
    half_height = height // 2
    
    boxes = []
    # Recursive subdivision for each quadrant
    boxes.extend(quadtree_compress_single(image, x, y, half_width, half_height, threshold, min_size))  # Top-left
    boxes.extend(quadtree_compress_single(image, x + half_width, y, half_width, half_height, threshold, min_size))  # Top-right
    boxes.extend(quadtree_compress_single(image, x, y + half_height, half_width, half_height, threshold, min_size))  # Bottom-left
    boxes.extend(quadtree_compress_single(image, x + half_width, y + half_height, half_width, half_height, threshold, min_size))  # Bottom-right

    return boxes

# Process a batch of images
def quadtree_bounding_boxes(image_batch, threshold=0.06, min_size=1):
    """
    Apply quadtree compression to each image in a batch independently and concatenate the results.
    
    Args:
        image_batch (Tensor): A batch of images of shape (batch_size, channels, height, width).
        threshold (float): Variance threshold for determining homogeneity.
        min_size (int): Minimum size of regions to split.
    
    Returns:
        Tensor: A tensor of shape (n, 5) where each row represents a bounding box with the format 
                [batch_index, x1, y1, x2, y2].
    """
    batch_size, channels, height, width = image_batch.shape
    all_bounding_boxes = []
    
    for i in range(batch_size):
        # Process each image independently
        image = image_batch[i]
        bounding_boxes = quadtree_compress_single(image, 0, 0, width, height, threshold, min_size)
        bounding_boxes = torch.tensor(bounding_boxes, device=image.device, dtype=torch.float32)
        
        # Rescale the bounding boxes to [0, 1]
        bounding_boxes[:, [0, 2]] /= width
        bounding_boxes[:, [1, 3]] /= height
        
        # Assertions to ensure the output is valid
        assert torch.all(bounding_boxes >= 0) and torch.all(bounding_boxes <= 1), "Bounding boxes should be in the range [0, 1]."
        assert not torch.isnan(bounding_boxes).any(), "Bounding boxes contain NaN values."
        
        # Add the batch index as the first column
        batch_indices = torch.full((bounding_boxes.shape[0], 1), i, device=image.device, dtype=torch.float32)
        bounding_boxes = torch.cat((batch_indices, bounding_boxes), dim=1)
        
        all_bounding_boxes.append(bounding_boxes)
    
    # Concatenate all bounding boxes into a single tensor
    all_bounding_boxes = torch.cat(all_bounding_boxes, dim=0)
    
    return all_bounding_boxes

# Map common color names to their BGR equivalents for OpenCV
COLOR_MAP = {
    'r': (0, 0, 255),  # Red
    'g': (0, 255, 0),  # Green
    'b': (255, 0, 0),  # Blue
    'y': (0, 255, 255),  # Yellow
    'c': (255, 255, 0),  # Cyan
    'm': (255, 0, 255),  # Magenta
    'orange': (0, 165, 255),  # Orange
    'purple': (128, 0, 128)  # Purple (approximated)
}

def draw_bounding_boxes(image_tensor, boxes, color_names=None, thickness=1):
    """
    Draws bounding boxes on the image tensor with different colors for each box.
    
    Parameters:
    - image_tensor (torch.Tensor): The input image tensor of shape (3, H, W).
    - boxes (torch.IntTensor): List of bounding box coordinates, each represented as (x_min, y_min, x_max, y_max).
    - color_names (List[str]): List of color names (e.g., 'r', 'g', 'b', 'orange', etc.) to cycle through.
    - thickness (int): Thickness of the bounding box lines.
    
    Returns:
    - torch.Tensor: Image tensor with bounding boxes drawn, of shape (3, H, W).
    """
    if color_names is None:
        color_names = ['r']  # Default color list
    
    # Convert color names to BGR tuples using the COLOR_MAP dictionary
    colors = [COLOR_MAP[color] for color in color_names]

    # Convert tensor to numpy array (H, W, 3) and transpose channels
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)  # Assuming image tensor is normalized between 0 and 1

    # Convert image from RGB (default in PyTorch) to BGR (used in OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Draw each bounding box on the image, cycling through the colors
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i].int().tolist()
        color = colors[i % len(colors)]  # Cycle through the color list
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, thickness)

    # Convert back to RGB format (from BGR used by OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Convert numpy array back to tensor and normalize to [0, 1]
    image_tensor_with_boxes = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float() / 255.0

    return image_tensor_with_boxes

def draw_bounding_boxes_batch(image_tensors, batch_boxes, color_names=None, thickness=1):
    """
    Processes a batch of images by drawing bounding boxes on each image.
    
    Parameters:
    - image_tensors (torch.Tensor): A batch of image tensors, shape (B, 3, H, W) where B is the batch size.
    - batch_boxes (List[torch.FloatTensor]): A list of bounding boxes, where each item corresponds
      to the bounding boxes for a particular image.
    - color_names (List[str]): List of color names (optional).
    - thickness (int): Thickness of the bounding box lines.
    
    Returns:
    - torch.Tensor: A batch of images with bounding boxes drawn, shape (B, 3, H, W).
    """
    batch_size, _, H, W = image_tensors.shape
    batch_boxes[:, [1, 3]] *= W
    batch_boxes[:, [2, 4]] *= H
    batch_boxes = batch_boxes.int()
    
    # List to store processed images with bounding boxes
    processed_images = []
    
    for i in range(batch_size):
        image_tensor = image_tensors[i]  # Get the i-th image in the batch
        boxes = batch_boxes[batch_boxes[:,0]==i][:, 1:]  # Get the corresponding bounding boxes for this image
        
        # Draw bounding boxes for this image
        processed_image = draw_bounding_boxes(image_tensor, boxes, color_names, thickness)
        processed_images.append(processed_image)
    
    # Stack the processed images back into a single batch tensor
    return torch.stack(processed_images)


if __name__ == '__main__':
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='QuadTree Image Processing')
    parser.add_argument('--lq_path', type=str, required=True, 
                       help='Path to low quality input image')
    parser.add_argument('--scale_factor', type=int, required=True, 
                       help='scale factor for upsampled image')
    parser.add_argument('--threshold', type=float, required=True, 
                       help='threshold for quadtree compression')
    args = parser.parse_args()

    image_lq = Image.open(args.lq_path).convert('L')
    
    # Transform to tensor and normalize to [0, 1]
    transform_lq = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
    transform_hq = transforms.Compose([
            transforms.ToTensor(),
        ])
    image_lq_tensor = transform_lq(image_lq).unsqueeze(0) # (1,C,H,W)
    upsampled_lq_tensor = F.interpolate(
        (image_lq_tensor * 0.5 + 0.5),  
        scale_factor=args.scale_factor,
        mode='bicubic',
        align_corners=False
    ).clamp(0, 1)  # [1, C, 4H, 4W]
    
    boxes = quadtree_bounding_boxes(image_lq_tensor, threshold=args.threshold, min_size=1)
    processed_image = upsampled_lq_tensor

    # Convert tensor to a PIL Image and save as PNG
    processed_image_np = (processed_image.squeeze(0).permute(1, 2, 0).numpy()* 255).astype(np.uint8)  # Convert to (H, W, C) format
    processed_image_pil = Image.fromarray(processed_image_np)

    # Save the processed image
    output_path = 'processed_image.png'  # Specify the output path
    processed_image_pil.save(output_path)
    print(f"Processed image saved at: {output_path}")