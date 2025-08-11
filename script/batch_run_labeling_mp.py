import os
import subprocess
import glob
from p_tqdm import p_map
import sys
import time

# Keep the configuration from the original script
script_dir = "/exp/sbnd/app/users/yuhw/dl-clus/script"
# Add script_dir to Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
# Configuration from lines 17-21 of the original script
input_path = "/exp/sbnd/app/users/yuhw/dl-clus/sample/20250618/"
output_path = "/exp/sbnd/app/users/yuhw/dl-clus/sample/20250618/"
job_batchid = 77451011
start_job = 0
end_job = 99

def verify_files_exist(job_folder, prefix):
    """Check if files with the given prefix exist in the job folder"""
    pattern = os.path.join(job_folder, f"{prefix}-*.npz")
    files = glob.glob(pattern)
    return len(files) > 0

def get_max_event(job_folder, prefix):
    """Find the maximum event number for files with the given prefix"""
    max_event = -1
    for npz_file in glob.glob(os.path.join(job_folder, f"{prefix}-*.npz")):
        filename = os.path.basename(npz_file)
        try:
            event_num = int(filename.split('-')[2].split('.')[0])
            if event_num > max_event:
                max_event = event_num
        except (ValueError, IndexError):
            continue
    return max_event

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
    
    # Check if both apa0 and apa1 files exist
    has_apa0 = verify_files_exist(job_folder, "rec-apa0")
    has_apa1 = verify_files_exist(job_folder, "rec-apa1")
    
    if not has_apa0 and not has_apa1:
        print(f"No rec-apa*-*.npz files found in {job_folder}. Skipping labeling.")
        return
    
    # Get max event for each APA
    max_event_apa0 = get_max_event(job_folder, "rec-apa0") if has_apa0 else -1
    max_event_apa1 = get_max_event(job_folder, "rec-apa1") if has_apa1 else -1
    
    print(f"Found files in {job_folder_name}: APA0={has_apa0}(max={max_event_apa0}), APA1={has_apa1}(max={max_event_apa1})")
    
    # Save current directory
    current_dir = os.getcwd()
    
    try:
        # Change to output folder
        os.chdir(out_folder)
        
        # Create symbolic link to labeling.py
        subprocess.run(["ln", "-sf", os.path.join(script_dir, "labeling.py"), "."])
        
        # Process each APA separately to better capture errors
        if has_apa1:
            print(f"Processing APA1 for {job_folder_name}")
            with open("log_apa1", "w") as log:
                proc = subprocess.run(
                    ["python", "labeling.py", 
                     "--tru-prefix", "tru-apa1", 
                     "--rec-prefix", "rec-apa1", 
                     "--out-prefix", "rec-lab-apa1", 
                     "--entries", f"0-{max_event_apa1}"],
                    stdout=log, stderr=subprocess.STDOUT, text=True
                )
                if proc.returncode != 0:
                    print(f"ERROR: APA1 processing failed for {job_folder_name} with return code {proc.returncode}")
            
            # Add small delay to prevent resource contention
            time.sleep(0.5)
        
        if has_apa0:
            print(f"Processing APA0 for {job_folder_name}")
            with open("log_apa0", "w") as log:
                proc = subprocess.run(
                    ["python", "labeling.py", 
                     "--tru-prefix", "tru-apa0", 
                     "--rec-prefix", "rec-apa0", 
                     "--out-prefix", "rec-lab-apa0", 
                     "--entries", f"0-{max_event_apa0}"],
                    stdout=log, stderr=subprocess.STDOUT, text=True
                )
                if proc.returncode != 0:
                    print(f"ERROR: APA0 processing failed for {job_folder_name} with return code {proc.returncode}")
        
        # Remove the symbolic link
        if os.path.exists("labeling.py"):
            os.remove("labeling.py")
    finally:
        # Return to original directory
        os.chdir(current_dir)

def main():
    # Create output path if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate list of job IDs to process
    job_ids = list(range(start_job, end_job + 1))
    
    # Process job folders in parallel with fewer processes to reduce contention
    p_map(process_job_folder, job_ids, num_cpus=30)
    
    print("All job folders processed successfully!")

if __name__ == "__main__":
    main()