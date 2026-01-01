import math
import pandas as pd
import numpy as np
from nipype.interfaces import afni, fsl
import nibabel as nib
import matplotlib.pyplot as plt
import os, datetime
import argparse
import sys
import shutil
import glob
import subprocess
from pathlib import Path
import re
import getpass


#-------------------------------------------------------------------------------------------------------------------------#
#                                             Defining Functions
#-------------------------------------------------------------------------------------------------------------------------#

# Function for Motion Correction using nypipe: AFNI's 3dvolreg
def smooth_movavg(in_file, out_file, win_sec_duration, tr):
   
  # inp, outp, win_sec_str, tr_str = sys.argv[1:5]
  inp = in_file
  outp = out_file
  win_sec_str = win_sec_duration
  tr_str = tr

  win_sec = float(win_sec_str)
  TR = float(tr_str)
  win = max(1, int(round(win_sec / TR)))

  def moving_average_1d(x, win):
      k = np.ones(win, dtype=float) / win
      xpad = np.pad(x, (win//2, win-1-win//2), mode='edge')  # reduce edge shrinkage
      return np.convolve(xpad, k, mode='valid')

  img = nib.load(inp)
  data = img.get_fdata()   # X,Y,Z,T
  T = data.shape[-1]
  flat = data.reshape(-1, T)
  sm = np.vstack([moving_average_1d(ts, win) for ts in flat]).reshape(data.shape)

  nib.Nifti1Image(sm, img.affine, img.header).to_filename(outp)
  print(f"Wrote: {outp}  (TR={TR}s, window={win_sec}s => {win} vols)")

def process_raw_data(in_path, scan_number):
  raw_data_path = os.path.join(in_path, scan_number)
  print_statement(f"Raw Data Path: {raw_data_path}", bcolors.NOTIFICATION)

  params = func_param_extract(raw_data_path, export_env=True)
  SequenceName = params["SequenceName"]
  
  analysed_folder_name = os.path.join(analysed_path, str(scan_number) + SequenceName)
  print_statement(f"Analysed Data Path: {analysed_folder_name}", bcolors.NOTIFICATION)
  if os.path.exists(analysed_folder_name):
    print_statement("Analysed Data Path exists.", bcolors.OKGREEN)
  else:
    os.makedirs(analysed_folder_name)
  
  os.chdir(analysed_folder_name)
  cwd = os.getcwd()

  if os.path.exists(os.path.join(analysed_folder_name, "G1_cp_resampled.nii.gz")):
    print_statement("NIFTI file already exists. Skipping conversion.", bcolors.OKGREEN)
  else:
    bruker_to_nifti(in_path, scan_number) 

def bruker_to_nifti(in_path, scan_number):
    
    scan_dir = os.path.join(in_path, scan_number)
    method_file = os.path.join(scan_dir, "method")

    # ---------- 1) Run brkraw tonii ----------
    cmd = ["brkraw", "tonii", f"{in_path}/", "-s", str(scan_number)]
    subprocess.run(cmd, check=True)

    # ---------- 2) Detect echo count in "method" ----------
    if "PVM_NEchoImages" in open(method_file).read():
        # Extract number of echoes using awk logic in Python
        with open(method_file) as f:
            for line in f:
                if "PVM_NEchoImages=" in line:
                    # Extract numeric part exactly like Bash substring 20..21
                    echo_str = line.split("=")[1].strip()
                    NoOfEchoImages = int(echo_str)
                    break

        # ---------- 3) If single echo ----------
        if NoOfEchoImages == 1:
            src_files = glob.glob(f"*{scan_number}*")
            for src in src_files:
                shutil.copy(src, "G1_cp.nii.gz")

        # ---------- 4) Multi-echo: merge then copy ----------
        else:
            merged_file = f"{scan_number}_combined_images"
            src_files = glob.glob(f"*{scan_number}*")
            # fslmerge -t combined.nii.gz file1 file2 file3 ...
            subprocess.run(["fslmerge", "-t", merged_file] + src_files, check=True)
            shutil.copy(f"{merged_file}.nii.gz", "G1_cp.nii.gz")

    else:
        # ---------- 5) No echo metadata ----------
        src_files = glob.glob(f"*{scan_number}*")
        for src in src_files:
            shutil.copy(src, "G1_cp.nii.gz")

    print(f"{bcolors.NOTIFICATION}Fixing orientation to LPI{bcolors.ENDC}")

    # ---------- 6) Fix orientation to LPI using 3dresample ----------
    resample = afni.Resample()
    resample.inputs.in_file = "G1_cp.nii.gz"
    resample.inputs.out_file = "G1_cp_resampled.nii.gz"
    resample.inputs.orientation = "LPI"
    resample.run()

    # ---------- 7) Save NIfTI header info ----------
    with open("NIFTI_file_header_info.txt", "w") as out:
        subprocess.run(["fslhd", "G1_cp_resampled.nii.gz"], stdout=out, check=True)

    print_statement(f"[OK] Bruker → NIFTI workflow completed.", bcolors.OKGREEN)

def extract_middle_volume(in_file, reference_vol, out_file, size):
  extract_vol = fsl.ExtractROI()
  extract_vol.inputs.in_file=in_file 
  extract_vol.inputs.t_min=reference_vol 
  extract_vol.inputs.t_size=size 
  extract_vol.inputs.roi_file=out_file
  extract_vol.run()

  print("[OK] Intended Volumes extracted.")
  return out_file

def motion_correction(reference_vol, input_vol, output_prefix):

    # ---------- 1) 3dvolreg ----------
    
    volreg = afni.Volreg()  
    volreg.inputs.in_file = input_vol
    volreg.inputs.basefile = reference_vol
    volreg.inputs.out_file = f"{output_prefix}.nii.gz"
    volreg.inputs.oned_file = "motion.1D"
    volreg.inputs.args = '-linear'
    volreg.inputs.oned_matrix_save = "mats"
    volreg.inputs.oned_matrix_save = "rmsabs.1D"
    volreg.inputs.verbose = True
    volreg.run()

    print("[INFO] Running 3dvolreg…")
    return output_prefix

def plot_motion_parameters(input_file):
    # ---------- 4) Plot motion parameters ----------
    print("[INFO] Creating motion plots…")

    # Translation plots
    data = np.loadtxt(input_file)

    # If your file HAS a header, use:
    # data = np.loadtxt("your_file.1D", comments="#")

    # X-axis (row index / timepoints)
    x = np.arange(data.shape[0])

    # -------- Plot 1: first 3 columns --------
    plt.figure(figsize=(8, 4))
    for i in range(3):
        plt.plot(x, data[:, i], label=f"Column {i+1}")

    plt.title("Rotation")
    plt.xlabel("Volume Number")
    plt.ylabel("Rotation in degrees")
    plt.legend(["Pitch (x)", "Roll (y)", "Yaw (z)"])
    plt.tight_layout()
    plt.savefig("motion_rotations.svg", dpi=1200)
    # -------- Plot 2: next 3 columns --------
    plt.figure(figsize=(8, 4))
    for i in range(3, 6):
        plt.plot(x, data[:, i], label=f"Column {i+1}")

    plt.title("Translation")
    plt.xlabel("Volume Number")
    plt.ylabel("Translation in mm")
    plt.legend(["Read (x)", "Phase (y)", "Slice (z)"])
    plt.tight_layout()
    plt.savefig("motion_translations.svg", dpi=1200)

def compute_mean_range(input_file, prefix, start_idx, end_idx):
    
    afni_cmd = ["3dTstat", "-mean", "-prefix", prefix, f"{input_file}[{start_idx}..{end_idx}]"]

    print("[INFO] Running:", " ".join(afni_cmd))

    subprocess.run(afni_cmd, check=True)
    print_statement("[OK] Mean baseline image saved.", bcolors.OKGREEN)

def masking_file(input_file, mask_file, output_file):
  
  math = fsl.maths.ApplyMask()
  math.inputs.in_file = input_file
  math.inputs.mask_file = mask_file
  math.inputs.out_file = output_file

  math.run()

  print_statement(f"[OK] Masked file saved → {output_file}", bcolors.OKGREEN)
  return output_file

def tSNR(input_file, output_file, reference_vol, size):
  
  extract_middle_volume(input_file, reference_vol, "extracted_ts.nii.gz", size)

  mean = fsl.maths.MeanImage()
  mean.inputs.in_file = "extracted_ts.nii.gz"
  mean.inputs.out_file = "mean_image.nii.gz"
  mean.run()

  std = fsl.maths.StdImage()
  std.inputs.in_file = "extracted_ts.nii.gz"
  std.inputs.out_file = "std_image.nii.gz"
  std.run()

  tSNR = fsl.maths.BinaryMaths()
  tSNR.inputs.in_file = "mean_image.nii.gz"
  tSNR.inputs.operand_file = "std_image.nii.gz"
  tSNR.inputs.operation = "div"
  tSNR.inputs.out_file = output_file
  tSNR.run()

  print_statement(f"[OK] tSNR file saved → {output_file}", bcolors.OKGREEN)
  return output_file

def spatial_smoothing(input_file, output_file, fwhm):
  
  smooth = fsl.maths.IsotropicSmooth()
  smooth.inputs.in_file = input_file
  smooth.inputs.out_file = output_file
  # smooth.inputs.fwhm = fwhm
  smooth.inputs.sigma = fwhm / 2.3548  # Convert FWHM to sigma

  smooth.run()

  print_statement(f"[OK] Spatially smoothed file saved → {output_file}", bcolors.OKGREEN)
  return output_file

def signal_change_map(signal_file, baseline_file, output_file):
   
  tmp_sub = "tmp_signal_minus_baseline.nii.gz"
  tmp_div = "tmp_psc_raw.nii.gz"

  sub = fsl.BinaryMaths()
  sub.inputs.in_file = signal_file
  sub.inputs.operand_file = baseline_file
  sub.inputs.operation = "sub"
  sub.inputs.out_file = tmp_sub
  sub.run()

  div = fsl.BinaryMaths()
  div.inputs.in_file = tmp_sub
  div.inputs.operand_file = baseline_file
  div.inputs.operation = "div"
  div.inputs.out_file = tmp_div
  div.run()

  mul = fsl.BinaryMaths()
  mul.inputs.in_file = tmp_div
  mul.inputs.operation = "mul"
  mul.inputs.operand_value = 100
  mul.inputs.out_file = output_file
  mul.run()

  os.remove(tmp_sub)
  os.remove(tmp_div)

  print_statement(f"[OK] Percent Signal Change Map saved → {output_file}", bcolors.OKGREEN)
  return output_file

def coregistration_afni(input_file1=None, input_file2=None, reference_file=None, output_file1=None, output_file2=None, estimate_affine=True, apply_affine=True, affine_mat="mean_func_struct_aligned.aff12.1D"):

  results = {}

  # -------- STEP 1: Estimate affine --------
  if estimate_affine:
      if output_file1 is None:
          raise ValueError("output_file1 must be provided when estimate_affine=True")
      if input_file1 is None:
          raise ValueError("input_file1 must be provided when estimate_affine=True")

      coreg_wo_affine = afni.Allineate()
      coreg_wo_affine.inputs.in_file = input_file1
      coreg_wo_affine.inputs.reference = reference_file
      coreg_wo_affine.inputs.out_matrix = affine_mat
      coreg_wo_affine.inputs.cost = "crU"
      coreg_wo_affine.inputs.two_pass = True
      coreg_wo_affine.inputs.verbose = True
      coreg_wo_affine.inputs.out_file = output_file1
      coreg_wo_affine.inputs.out_param_file = "params.1D"
      coreg_wo_affine.run()

      print(f"[OK] Affine estimated and saved → {affine_mat}")
      print(f"[OK] Coregistered image (step 1) → {output_file1}")

      results["step1"] = output_file1

  # -------- STEP 2: Apply affine --------
  if apply_affine:
      if output_file2 is None:
          raise ValueError("output_file2 must be provided when apply_affine=True")
      if input_file2 is None:
          raise ValueError("input_file2 must be provided when apply_affine=True")


      coreg_with_affine = afni.Allineate()
      coreg_with_affine.inputs.in_file = input_file2
      coreg_with_affine.inputs.reference = reference_file
      coreg_with_affine.inputs.in_matrix = affine_mat
      coreg_with_affine.inputs.master = reference_file
      coreg_with_affine.inputs.verbose = True
      coreg_with_affine.inputs.final_interpolation = "linear"
      coreg_with_affine.inputs.out_file = output_file2
      coreg_with_affine.run()

      print_statement(f"[OK] Affine applied → {output_file2}", bcolors.OKGREEN)
      results["step2"] = output_file2

  return results

def time_course_extraction(roi_file, func_file, output_file):
   
    ts = fsl.ImageMeants()
    ts.inputs.in_file = func_file
    ts.inputs.mask = roi_file
    ts.inputs.out_file = output_file

    ts.run()

def func_param_extract(scan_dir, export_env=True):

    scan_dir = Path(scan_dir)
    acqp_file = scan_dir / "acqp"
    method_file = scan_dir / "method"

    if not acqp_file.exists() or not method_file.exists():
        raise FileNotFoundError("acqp or method file not found")

    # -----------------------------
    # Read files
    # -----------------------------
    acqp_text = acqp_file.read_text()
    method_text = method_file.read_text()

    # -----------------------------
    # Sequence name (ACQ_protocol_name)
    # -----------------------------
    seq_match = re.search(
        r"ACQ_protocol_name=\(\s*64\s*\)\s*\n\s*<([^>]+)>",
        acqp_text
    )
    SequenceName = seq_match.group(1) if seq_match else None

    # -----------------------------
    # Extract numeric parameters
    # -----------------------------
    def get_value(pattern, text, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    NoOfRepetitions = get_value(r"##\$PVM_NRepetitions=\s*(\d+)", method_text)
    TotalScanTime = get_value(r"##\$PVM_ScanTime=\s*(\d+)", method_text)

    Baseline_TRs = get_value(r"PreBaselineNum=\s*(\d+)", method_text)
    StimOn_TRs = get_value(r"StimNum=\s*(\d+)", method_text)
    StimOff_TRs = get_value(r"InterStimNum=\s*(\d+)", method_text)
    NoOfEpochs = get_value(r"NEpochs=\s*(\d+)", method_text)

    # -----------------------------
    # Derived values
    # -----------------------------
    VolTR_msec = None
    VolTR = None
    MiddleVolume = None

    if NoOfRepetitions and TotalScanTime:
        VolTR_msec = TotalScanTime / NoOfRepetitions
        VolTR = VolTR_msec / 1000
        MiddleVolume = NoOfRepetitions / 2

    # -----------------------------
    # Pack results
    # -----------------------------
    params = {
        "SequenceName": SequenceName,
        "NoOfRepetitions": NoOfRepetitions,
        "TotalScanTime": TotalScanTime,
        "VolTR_msec": VolTR_msec,
        "VolTR": VolTR,
        "Baseline_TRs": Baseline_TRs,
        "StimOn_TRs": StimOn_TRs,
        "StimOff_TRs": StimOff_TRs,
        "NoOfEpochs": NoOfEpochs,
        "MiddleVolume": MiddleVolume,
    }

    # -----------------------------
    # Export to environment (optional)
    # -----------------------------
    if export_env:
        for k, v in params.items():
            if v is not None:
                os.environ[k] = str(v)

    return params

def print_header(message, color):
    line = "*" * 134   # same width everywhere
    width = len(line)

    print()
    print(f"{color}{line}{bcolors.ENDC}")
    print(f"{color}{message.center(width)}{bcolors.ENDC}")
    print(f"{color}{line}{bcolors.ENDC}")
    print()

def print_statement(message, color):
    print(f"{color}{message}{bcolors.ENDC}")



# -------------------------------------------------------------------------------------------------------------------------#
#                                             Execution of Main Script
#-------------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

  ap = argparse.ArgumentParser(description="Input Data, Output data and Scan Numbers")
  # ap.add_argument("root")
  ap.add_argument("--in_path", type=str, required=True)
  ap.add_argument("--func", type=str, default=2)
  ap.add_argument("--struct", type=str, default=2)
  ap.add_argument("--win_dur", type=str, default=None, required=True)
  args = ap.parse_args()
  
  in_path = args.in_path
  func_scan_number = args.func
  struct_scan_number = args.struct
  win_dur = int(args.win_dur)

  analysed_path = Path(str(in_path).replace("/RawData/", "/AnalysedData/"))

  class bcolors:
      HEADER = '\033[95m'
      OKBLUE = '\033[94m'
      OKCYAN = '\033[96m'
      OKGREEN = '\033[92m'
      NOTIFICATION = '\033[93m'
      FAIL = '\033[91m'
      ENDC = '\033[0m'
      BOLD = '\033[1m'
      UNDERLINE = '\033[4m'

  print_header("Converting Bruker to NIFTI: Both Structural and Functional Data", bcolors.HEADER)

  process_raw_data(in_path, struct_scan_number)
  process_raw_data(in_path, func_scan_number)

  # Applying Motion Correction on raw functional data and plotting motion parameters
  
  print_header("Applying Motion Correction on raw functional data and plotting motion parameters", bcolors.HEADER)

  path_raw_func = os.path.join(in_path, func_scan_number)
  params = func_param_extract(path_raw_func, export_env=True)
  SequenceName = params["SequenceName"]
  tr = int(params["VolTR"])
  n_vols = params["NoOfRepetitions"]
  middle_vol = str(int(n_vols / 2))

  extract_middle_volume("G1_cp_resampled.nii.gz", int(middle_vol), "middle_vol.nii.gz", 1)

  if os.path.exists("mc_func.nii.gz"):
    print(f"{bcolors.OKGREEN}Motion Corrected functional data exists. Skipping motion correction.{bcolors.ENDC}")
  else:
    motion_correction("middle_vol.nii.gz", input_vol="G1_cp_resampled.nii.gz", output_prefix="mc_func")

  plot_motion_parameters("motion.1D")

  #Creating a mask to be applied on functional data using the mean baseline image
  if os.path.exists("mask_mean_mc_func.nii.gz"):
    print(f"{bcolors.OKGREEN}Mask Image exists.{bcolors.ENDC}")
  else:
    print(f"{bcolors.FAIL}Mask Image does not exist. Please create the mask and save it as mask_mean_mc_func.nii.gz{bcolors.ENDC}")
    subprocess.run(["fsleyes", "middle_vol.nii.gz"])

  if os.path.exists("mask_mean_mc_func_cannulas.nii.gz"):
    print(f"{bcolors.OKGREEN}Mask Image including cannulas exist.{bcolors.ENDC}")
  else:
    print(f"{bcolors.FAIL}Mask Image does not exist. Please create the mask that also includes cannulas and save it as mask_mean_mc_func_cannulas.nii.gz.{bcolors.ENDC}")
    shutil.copyfile("mask_mean_mc_func.nii.gz", "mask_mean_mc_func_cannulas.nii.gz")
    subprocess.run(["fsleyes", "mean_image.nii.gz" , "mask_mean_mc_func_cannulas.nii.gz"])
  
  # Setting up analysis directory
  print_header("Setting up Analysis Directory", bcolors.HEADER)

  path_current_analysis = os.path.join(analysed_path, str(func_scan_number) + SequenceName, datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S') + "_" + getpass.getuser())
  os.makedirs(path_current_analysis)
  shutil.copy(os.path.join(os.getcwd(), "mc_func.nii.gz"), os.path.join(path_current_analysis))
  shutil.copy(os.path.join(os.getcwd(), "mask_mean_mc_func.nii.gz"), os.path.join(path_current_analysis))
  os.chdir(path_current_analysis)
  cwd = os.getcwd()

  #Applying temporal smoothing to the motion corrected functional data to see the temporal signatures
  print_header("Applying temporal smoothing to the motion corrected functional data to see the temporal signatures", bcolors.HEADER)
  smooth_movavg("mc_func.nii.gz", "temporal_smoothed_mc_func.nii.gz", 60, 1.0)

  # Opening fsleyes to view the temporally smoothed motion corrected functional data
  print_statement("Choose your baseline and signal volumes from the temporally smoothed motion corrected functional data.", bcolors.NOTIFICATION)
  subprocess.run(["fsleyes", "temporal_smoothed_mc_func.nii.gz"])

  #Choosing signal and baseline volume indices from the temporally smoothed motion corrected functional data
  base_start = 400
  sig_start  = 1900
  # base_start = int(input("Enter baseline start Volume index: "))
  # sig_start  = int(input("Enter signal start Volume index: "))
  base_end   = base_start + win_dur
  sig_end   = sig_start + win_dur

  compute_mean_range(input_file="temporal_smoothed_mc_func.nii.gz", prefix=f"mean_baseline_image_{base_start}_to_{base_end}.nii.gz", start_idx=base_start, end_idx=base_end)
  os.remove(f"mean_baseline_image_{base_start}_to_{base_end}.nii.gz")

  #Masking temporally smoothed motion corrected functional data using the created mask
  print_header("Masking temporally smoothed motion corrected functional data using the created mask", bcolors.HEADER)
  masking_file(input_file="temporal_smoothed_mc_func.nii.gz", mask_file="mask_mean_mc_func.nii.gz", output_file="cleaned_mc_func.nii.gz") #creating cleaned motion corrected functional data from temporal smoothed data for further processing
  masking_file(input_file="mc_func.nii.gz", mask_file="mask_mean_mc_func.nii.gz", output_file="raw_cleaned_mc_func.nii.gz") #creating cleaned motion corrected functional data from raw data for different processing

  #Estimating tSNR using the cleaned motion corrected functional data
  print_header("Estimating tSNR using the cleaned motion corrected functional data", bcolors.HEADER)
  tSNR(input_file="cleaned_mc_func.nii.gz", reference_vol=100, output_file="tSNR_mc_func.nii.gz", size=400)

  # Applying isotropic spatial smoothing on cleaned motion corrected functional data
  print_header("Applying isotropic spatial smoothing on cleaned_mc_func.nii.gz", bcolors.HEADER)
  spatial_smoothing('cleaned_mc_func.nii.gz', 'smoothed_cleaned_mc_func.nii.gz', float(0.7))

  #Generating Signal Change Map and Signal Change Time Series
  print_header("Generating Signal Change Map and Signal Change Time Series", bcolors.HEADER)

  compute_mean_range(input_file="smoothed_cleaned_mc_func.nii.gz", prefix=f"mean_baseline_image_{base_start}_to_{base_end}.nii.gz", start_idx=base_start, end_idx=base_end)
  compute_mean_range(input_file="smoothed_cleaned_mc_func.nii.gz", prefix=f"mean_signal_image_{sig_start}_to_{sig_end}.nii.gz", start_idx=sig_start, end_idx=sig_end)

  signal_change_map(f"mean_signal_image_{sig_start}_to_{sig_end}.nii.gz", f"mean_baseline_image_{base_start}_to_{base_end}.nii.gz", "tmp_signal_change_map.nii.gz")
  masking_file(input_file="tmp_signal_change_map.nii.gz", mask_file="mask_mean_mc_func.nii.gz", output_file="cleaned_SCM_func.nii.gz") #cleaning the signal change map
  os.remove("tmp_signal_change_map.nii.gz")
  
 # Creating time series of signal change

  print_header("Creating time series of signal change", bcolors.HEADER)

  compute_mean_range(input_file="smoothed_cleaned_mc_func.nii.gz", prefix=f"mean_sm_baseline_image_{base_start}_to_{base_end}.nii.gz", start_idx=base_start, end_idx=base_end)
  signal_change_map("smoothed_cleaned_mc_func.nii.gz", f"mean_sm_baseline_image_{base_start}_to_{base_end}.nii.gz", "tmp_signal_change_time_series.nii.gz")
  masking_file("tmp_signal_change_time_series.nii.gz", "mask_mean_mc_func.nii.gz", "norm_cleaned_mc_func.nii.gz")
  os.remove("tmp_signal_change_time_series.nii.gz")

  #Cleaning the structural image by masking it with a manually created mask
  
  print_header("Cleaning the structural image by manually creating mask", bcolors.HEADER)
  params_struct = func_param_extract(os.path.join(in_path, struct_scan_number), export_env=True)
  seq_name_struct = params_struct["SequenceName"]
  struct_coreg_dir = os.path.join(analysed_path, str(struct_scan_number) + seq_name_struct)

  structural_file_for_coregistration = os.path.join(struct_coreg_dir, "cleaned_anatomy.nii.gz")
  if os.path.exists(os.path.join(struct_coreg_dir, "cleaned_anatomy.nii.gz")):
    print_statement("Structural Image for Coregistration exists.", bcolors.OKGREEN)
  else:
    print_statement("Please create a mask for the structural image and save it as mask_anatomy.nii.gz", bcolors.NOTIFICATION)
    subprocess.run(["fsleyes", os.path.join(struct_coreg_dir, "G1_cp_resampled.nii.gz")])
    masking_file(os.path.join(struct_coreg_dir, "G1_cp_resampled.nii.gz"), os.path.join(struct_coreg_dir, "mask_anatomy.nii.gz"), structural_file_for_coregistration)

  masking_file(os.path.join(analysed_path, str(func_scan_number) + SequenceName, "middle_vol.nii.gz"), os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mask_mean_mc_func_cannulas.nii.gz"), "cleaned_mean_mc_func_cannulas.nii.gz")
  masking_file(os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mc_func.nii.gz"), os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mask_mean_mc_func_cannulas.nii.gz"), "cleaned_mc_func_cannulas.nii.gz")
  
  #Coregistering functional time series and functional signal change map to structural image
  print_header("Coregistering functional time series and functional signal change map to structural image", bcolors.HEADER)
  affine_matrix_file = ("mean_func_struct_aligned.aff12.1D")
  if os.path.exists(affine_matrix_file):
    print_statement("Affine Matrix to coregister Signal Change Map exists.", bcolors.OKGREEN)
  else:
    print_statement("Estimating Affine Matrix to coregister Signal Change Map.", bcolors.NOTIFICATION)
    coregistration_afni(input_file1="cleaned_mean_mc_func_cannulas.nii.gz", input_file2="cleaned_SCM_func.nii.gz", reference_file= structural_file_for_coregistration, output_file1="mean_func_struct_aligned.nii.gz", output_file2="signal_change_map_coregistered_structural_space.nii.gz", estimate_affine=True, apply_affine=True, affine_mat="mean_func_struct_aligned.aff12.1D")

  #Coregistering functional time series and generating signal change map from coregistered data

  print_header("Coregistering functional time series and generating signal change map from coregistered data", bcolors.HEADER)
  coregistration_afni(input_file1=None, input_file2="cleaned_mc_func_cannulas.nii.gz", reference_file= structural_file_for_coregistration, output_file1=None, output_file2="fMRI_coregistered_to_struct.nii.gz", estimate_affine=False, apply_affine=True, affine_mat="mean_func_struct_aligned.aff12.1D")

  mean = fsl.maths.MeanImage()
  mean.inputs.in_file = "fMRI_coregistered_to_struct.nii.gz"
  mean.inputs.out_file = "mean_fMRI_coregistered_to_struct.nii.gz"
  mean.run()

# spatial_smoothing("fMRI_coregistered_to_struct.nii.gz", "sm_fMRI_coregistered_to_struct.nii.gz", fwhm=0.12)

  if os.path.exists(os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mask_mean_fMRI_coregistered_to_struct.nii.gz")):
    print("Mask file mask_mean_fMRI_coregistered_to_struct.nii.gz exists.")
  else:
    print("Please create a mask and save it as mask_mean_fMRI_coregistered_to_struct.nii.gz")
    subprocess.run(["fsleyes", "mean_fMRI_coregistered_to_struct.nii.gz"])
    shutil.copyfile(os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mask_mean_fMRI_coregistered_to_struct.nii.gz"), "mask_mean_fMRI_coregistered_to_struct.nii.gz")

  masking_file("fMRI_coregistered_to_struct.nii.gz", os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mask_mean_fMRI_coregistered_to_struct.nii.gz"), "sm_fMRI_for_scm.nii.gz")

  compute_mean_range(input_file="sm_fMRI_for_scm.nii.gz", prefix=f"baseline_sm_fMRI_for_scm_{base_start}_to_{base_end}.nii.gz", start_idx=base_start, end_idx=base_end)
  compute_mean_range(input_file="sm_fMRI_for_scm.nii.gz", prefix=f"signal_sm_fMRI_for_scm_{sig_start}_to_{sig_end}.nii.gz", start_idx=sig_start, end_idx=sig_end)

  signal_change_map(f"signal_sm_fMRI_for_scm_{sig_start}_to_{sig_end}.nii.gz", f"baseline_sm_fMRI_for_scm_{base_start}_to_{base_end}.nii.gz", f"sm_coreg_func_Static_Map_{base_start}_to_{base_end}_and_{sig_start}_to_{sig_end}.nii.gz")
  masking_file(f"sm_coreg_func_Static_Map_{base_start}_to_{base_end}_and_{sig_start}_to_{sig_end}.nii.gz", os.path.join(analysed_path, str(func_scan_number) + SequenceName, "mask_mean_fMRI_coregistered_to_struct.nii.gz"), "cleaned_sm_scm_from_coregistered_ts.nii.gz")


  #Marking ROIs and saving time courses

  print_header("Marking ROIs and saving time courses", bcolors.HEADER)

  print_statement("Please create ROIs on the functional time series and save them in the following particular format:", bcolors.NOTIFICATION) 
  print_statement("roi_{what protein/aav is there}_{is it direct injection or aav}_{analyte injeted}_{hemisphere side}.nii.gz", bcolors.NOTIFICATION)
  print_statement("For Example: if GCaMP6f is directly injected in the left hemisphere and dopamine is injected in the right hemisphere following a viral injection, then the following ROIs should be created:", bcolors.NOTIFICATION) 
  print_statement("roi_GCaMP6f_direct_left.nii.gz or roi_dopamine_aav_right.nii.gz", bcolors.FAIL)  

  subprocess.run(["fsleyes", "mean_fMRI_coregistered_to_struct.nii.gz"])

  files_list = os.listdir(cwd)
  roi_for_psc = [file for file in files_list if file.startswith("roi_") and file.endswith(".nii.gz")]
  for files in roi_for_psc:
    # skip if the file isn't a text file
      roi_file = files
      print_statement(f"Extracting time course for ROI: {roi_file}", bcolors.NOTIFICATION)
      output_file = f"time_course_{roi_file.replace('.nii.gz', '.txt')}"
      time_course_extraction(roi_file, "fMRI_coregistered_to_struct.nii.gz", output_file)
      print_statement(f"[OK] Time course saved → {output_file}", bcolors.OKGREEN)

      #Creating Percent Signal Change graphs for each ROI
      id_arr = list(range(0, n_vols, tr))
      time_series = np.loadtxt(output_file)
      # baseline = np.mean(time_series[base_start:base_end])
      baseline = np.mean(time_series[base_start:base_end])
      psc = ((time_series - baseline) / baseline) * 100
      print_statement(f"[OK] Percent Signal Change calculated for ROI: {roi_file}", bcolors.OKGREEN)
      print("Time Series is:", psc)
      np.savetxt(f"PSC_time_series_{roi_file.replace('.nii.gz', '.txt')}", psc)
      plt.figure(figsize=(10, 5))
      plt.plot(id_arr, psc, label='Percent Signal Change')
      plt.axvspan(base_start, base_end, color='green', alpha=0.3, label='Baseline Period')
      plt.axvspan(sig_start, sig_end, color='blue', alpha=0.3, label='Signal Period')
      plt.title(f'Percent Signal Change Time Series for {roi_file}')
      plt.xlabel('Time Points (Volumes)')
      plt.ylabel('MRI Signal Change (%)')
      plt.legend()
      plt.tight_layout()
      graph_file = f"PSC_Time_Series_{roi_file.replace('.nii.gz', '.svg')}"
      plt.savefig(graph_file, dpi=1200)
      print_statement(f"[OK] Percent Signal Change graph saved → {graph_file}", bcolors.OKGREEN)   

            



