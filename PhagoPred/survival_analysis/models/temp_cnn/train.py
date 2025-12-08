from pytorch_tcn import TCN

def main():
    features = [
        '0',
        '1',
        '2',
        '3',
        ] 
    
    model_params = {
        'num_inputs': len(features),
        'output_projection': 4, # number of time bins
        'kernel_size': 4,
        'dilations': [1, 2, 4, 8, 16, 32, 64],
        'output_activation': 'softmax',
        'use_skip_connections': False,
        'causal': True
    }
    model = TCN(**model_params)