import os
import opensmile
import pandas as pd

# Set paths for Mac
base_dir = "/Users/sahilchauhan/Downloads/multimodaldetection"
config_file_path = os.path.join(base_dir, "Androids-Corpus/Androids.conf")
output_folder = os.path.join(base_dir, "Audio-Interview")
data_dirs = [
    os.path.join(base_dir, "Androids-Corpus/Interview-Task/audio/HC"),
    os.path.join(base_dir, "Androids-Corpus/Interview-Task/audio/PT")
]

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
    print(f"Output folder: {output_folder}")
    
    all_audio_files = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
            all_audio_files.extend(audio_files)
            print(f"Found {len(audio_files)} audio files in {os.path.basename(data_dir)}")
        else:
            print(f"Directory not found: {data_dir}")
    
    print(f"\nTotal audio files to process: {len(all_audio_files)}")
    
    for i, audio_file in enumerate(all_audio_files):
        print(f"Processing [{i+1}/{len(all_audio_files)}]: {os.path.basename(audio_file)}")
        extract_audio_features_opensmile(audio_file, output_folder)
    
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    main()
