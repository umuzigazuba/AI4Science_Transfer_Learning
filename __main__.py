import argparse
from .src.data_generation.generation_functions import generate_samples_specific_type

def main():

    parser = argparse.ArgumentParser(description = "Generate a certain number of samples for a specific spectral type.")

    parser.add_argument("spectral_type", type = str, 
                        help = "The spectral type to generate samples for.")
    
    parser.add_argument("number", type = int, 
                        help = "Number of samples to generate.")
    
    parser.add_argument("output_file_name", type = str,
                        help = "Name of the output file.")
    
    parser.add_argument("--data_dir", type = str, default = "data",
                        help = "Path to the data directory (default: 'data').")
    
    parser.add_argument("--active_bands", type = str, default = "ugirz",
                        help = "Active bands to use (default: 'ugirz').")
 
    args = parser.parse_args()
 
    generate_samples_specific_type(spectral_type = args.spectral_type, number = args.number, 
                                   output_file_name = args.output_file_name, data_dir = args.data_dir,
                                   active_bands = args.active_bands)

if __name__ == "__main__":

    main()
