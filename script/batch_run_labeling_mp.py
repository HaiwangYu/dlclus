import os
import subprocess
import glob
from p_tqdm import p_map
import sys

# Keep the configuration from the original script
script_dir = "/exp/sbnd/app/users/yuhw/dl-clus/script"
# Add script_dir to Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
# Configuration from lines 17-21 of the original script
input_path = "/exp/sbnd/app/users/yuhw/dl-clus/sample/20250618/"
output_path = "/exp/sbnd/app/users/yuhw/dl-clus/sample/20250618/"
job_batchid = 77451011
start_job = 2
end_job = 99

def process_job_folder(job_id):
    """Process a single job folder"""
    job_folder_name = f"{job_batchid}_{job_id}"
    job_folder = os.path.join(input_path, job_folder_name)
    
    # Check if the job folder exists
    if not os.path.isdir(job_folder):
        print(f"Warning: Job folder '{job_folder}' does not exist, skipping...")
        return
    
    print(f"Processing job folder: {job_folder_name}")
    
    # Create corresponding output folder
    out_folder = os.path.join(output_path, job_folder_name)
    os.makedirs(out_folder, exist_ok=True)
    
    # Find all files matching the pattern rec-apa0-*.npz
    max_event = -1
    for npz_file in glob.glob(os.path.join(job_folder, "rec-apa0-*.npz")):
        # Extract event number from filename
        filename = os.path.basename(npz_file)
        try:
            event_num = int(filename.split('-')[2].split('.')[0])
            if event_num > max_event:
                max_event = event_num
        except (ValueError, IndexError):
            continue
    
    print(f"Maximum event number in {job_folder_name}: {max_event}")
    
    # Run the run_labeling.sh script if we found valid files
    if max_event == -1:
        print(f"No rec-apa0-*.npz files found in {job_folder}. Skipping labeling.")
        return
    
    # Save current directory
    current_dir = os.getcwd()
    
    try:
        # Change to output folder
        os.chdir(out_folder)
        
        # Create symbolic link to labeling.py
        subprocess.run(["ln", "-sf", os.path.join(script_dir, "labeling.py"), "."])
        
        # Run the labeling script using bash explicitly
        labeling_script = os.path.join(script_dir, "run_labeling.sh")
        print(f"Running: bash {labeling_script} 0 {max_event}")
        
        # Fix: Use bash explicitly to run the shell script
        subprocess.run(["bash", labeling_script, "0", str(max_event)], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Remove the symbolic link
        os.remove("labeling.py")
    finally:
        # Return to original directory
        os.chdir(current_dir)

def main():
    # Create output path if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate list of job IDs to process
    job_ids = list(range(start_job, end_job + 1))
    
    # Process job folders in parallel
    p_map(process_job_folder, job_ids, num_cpus=100)
    
    print("All job folders processed successfully!")

if __name__ == "__main__":
    main()
