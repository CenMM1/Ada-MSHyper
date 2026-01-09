from MSA_FET import FeatureExtractionTool, get_default_config
from MSA_FET import run_dataset


def main():
    # initialize with default librosa config which only extracts audio features
    config_a = get_default_config('wav2vec')
    config_t = get_default_config('bert')
    config = {**config_a, **config_t}

    # print(config)
    # print(get_default_config('aligned'))
    


    run_dataset(
        config=config,
        dataset_dir="mosi/", 
        out_file="output/mosi_feature.pkl",
        num_workers=16
    )



if __name__ == "__main__":
    main()
    
