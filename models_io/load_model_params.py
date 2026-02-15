import torch
import re

def load_model_params(filepath):
    model_state = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Split into sections based on the start of a line bracket [layer.name]
    sections = re.split(r'^\[', content, flags=re.MULTILINE)

    for section in sections:
        if not section or ']' not in section:
            continue
            
        header_line = section.split('\n', 1)[0]
        body = section.split('\n', 1)[1] if '\n' in section else ""

        # 1. Extract Layer Name
        key = header_line.split(']')[0].strip()
        
        # 2. Identify Tags (TRANSPOSED)
        is_transposed = "[TRANSPOSED]" in header_line.upper()
        
        # 3. Extract Shape Metadata
        shape_match = re.search(r"shape=\[(\d+)(?:,\s*(\d+))?\]", header_line)
        
        # 4. Extract Weight/Bias values (ignoring numbers in the header)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", body)
        if not numbers:
            continue
            
        float_data = torch.tensor([float(n) for n in numbers])
        
        if shape_match:
            rows = int(shape_match.group(1))
            cols = int(shape_match.group(2)) if shape_match.group(2) else None
            
            if cols:
                # If the file says it is transposed, we must reshape 
                # and then transpose it back to PyTorch's (out, in) format
                tensor = float_data.reshape(rows, cols)
                if is_transposed:
                    tensor = tensor.T
                model_state[key] = tensor
            else:
                model_state[key] = float_data
        else:
            model_state[key] = float_data

    return model_state
