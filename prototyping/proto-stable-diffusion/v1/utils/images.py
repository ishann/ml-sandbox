from PIL import Image

def make_translucent(input_path, output_path, alpha_factor=0.5):
    """
    Reduces the opacity of a PNG image.
    
    Parameters:
    - input_path (str): Path to the input PNG file.
    - output_path (str): Path to save the modified image.
    - alpha_factor (float): Factor to reduce alpha (0.0 to 1.0).
    
    Example usage
    make_translucent("input.png", "output_translucent.png", alpha_factor=0.5)
    """
    img = Image.open(input_path).convert("RGBA")
    new_data = []

    for r, g, b, a in img.getdata():
        new_alpha = int(a * alpha_factor)
        new_data.append((r, g, b, new_alpha))

    img.putdata(new_data)
    img.save(output_path)

