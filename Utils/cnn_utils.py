import numpy as np

def get_cnn_output_size(input_size, kernel_size=(1,1), stride=(1,1), padding=(0,0)):
    """
    Calculate the output size of a CNN layer.
    
    Args:
        input_size (tuple): (H, W) input dimensions
        kernel_size (tuple): (Kh, Kw)
        stride (tuple): (Sh, Sw)
        padding (tuple): (Ph, Pw)
    
    Returns:
        tuple: (H_out, W_out)
    """
    H, W = input_size
    Kh, Kw = kernel_size
    Sh, Sw = stride
    Ph, Pw = padding

    #floor to account for uneven division
    H_out = np.floor(((H + 2*Ph - Kh) / Sh) + 1).astype(int) 
    W_out = np.floor(((W + 2*Pw - Kw) / Sw) + 1).astype(int)

    return (H_out, W_out)
