import os
import opensmile
import pandas as pd

# Set paths for Mac - extract features from PT (Patient/Depressed) folder
base_dir = "/Users/sahilchauhan/Downloads/multimodaldetection"
output_folder = os.path.join(base_dir, "Audio-Interview")
data_dir = os.path.join(base_dir, "Androids-Corpus/Interview-Task/audio/PT")

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_audio_features_opensmile(audio_file, output_folder):
    """Extract audio features using opensmile Python package."""
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}. Skipping.")
        return None

    try:
        # Use eGeMAPSv02 feature set (commonly used for emotion/speech analysis)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        
        features = smile.process_file(audio_file)
        
        # Create output filename
        base_name = os.path.basename(audio_file).replace('.wav', '_it_audio_features.csv')
        output_csv_file = os.path.join(output_folder, base_name)
        
        # Save features
        features.to_csv(output_csv_file)
        print(f"Extracted: {base_name}")
        return output_csv_file
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def main():
    print(f"Extracting features from PT (Patient) folder")
    print(f"Output folder: {output_folder}")
    
    # Just get first 10 files to balance with HC files
    audio_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')])[:10]
    print(f"Processing first 10 PT files for balanced demo")
    
    for i, audio_file in enumerate(audio_files):
        print(f"Processing [{i+1}/{len(audio_files)}]: {os.path.basename(audio_file)}")
        extract_audio_features_opensmile(audio_file, output_folder)
    
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    main()
