from MSA_FET import FeatureExtractionTool, get_default_config
from MSA_FET import run_dataset


def main():
    # initialize with default librosa config which only extracts audio features
    config_a = get_default_config('wav2vec')
    config_v = get_default_config('openface')
    config_t = get_default_config('bert')
    config = {**config_a, **config_v, **config_t}

    print(config)

    # run_dataset(
    #     config=config,
    #     dataset_dir="dataset/", 
    #     out_file="output/feature.pkl",
    #     num_workers=4
    # )



if __name__ == "__main__":
    main()
    
