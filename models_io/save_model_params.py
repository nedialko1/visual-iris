import datetime
import os

# --------------------------------
from pathlib import Path

# 1. Get the path of the current script file
script_path = Path(__file__).resolve()
# 2. Get the directory containing the script
script_dir = script_path.parent

# --------------------------------------------------------

def format_val(x):
    """12-char wide, 4 decimal precision, IEEE-flexible."""
    if abs(x) > 999999999999.9999:
        return f"{x:12.4E}"
    return f"{x:12.4f}"

def save_model_params(model_name, layers_dict, version=None,
                      out_dir="models_io/saved_model_pars"):
    # 3. Define the relative path to the subfolder and file
    relative_subfolder_path = Path(f"../{out_dir}")
    # 4. Combine them to get the absolute path to the data file
    absolute_file_path = script_dir / relative_subfolder_path

    if not os.path.exists(absolute_file_path):
        os.makedirs(absolute_file_path)

    if version is None:
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_path = os.path.join(absolute_file_path, f"{model_name}_v{version}.txt")

    with open(file_path, 'w') as f:
        f.write(f"# MODEL: {model_name} | VER: {version}\n")
        f.write("-" * 120 + "\n\n")

        for key, tensor in layers_dict.items():
            data = tensor.detach().cpu().numpy()
            shape = list(data.shape)
            is_transposed = False

            # Logic for 2D Matrices (M x N)
            if len(shape) == 2:
                m, n = shape
                # If wider than it is tall, transpose for better vertical flow
                if m < n:
                    data = data.T
                    shape = list(data.shape)
                    is_transposed = True
            
            # Header with explicit Metadata
            transpose_tag = " [TRANSPOSED]" if is_transposed else ""
            f.write(f"[{key}] shape={shape}{transpose_tag}\n")

            # Writing the table
            if len(shape) == 2:
                for row in data:
                    formatted_row = " ".join([format_val(v) for v in row])
                    f.write(f"  {formatted_row}\n")
            
            elif len(shape) == 1:
                # 1D Vectors (Biases) or higher dims (flattened): proceed similarly
                flat_data = data.flatten()
                formatted_row = " ".join([format_val(v) for v in flat_data])
                f.write(f"  {formatted_row}\n")

            else:    
                # For higher dims (flattened), pack them reasonably (say 8 per line)
                for i in range(0, len(flat_data), 8):
                    chunk = flat_data[i : i + 8]
                    f.write("  " + " ".join([format_val(v) for v in chunk]) + "\n")
            
            f.write("\n")

    print(f"Model parameters saved to: {file_path}")
    return file_path
