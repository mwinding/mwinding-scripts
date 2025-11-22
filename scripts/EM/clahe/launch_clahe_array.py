#!/usr/bin/env python3
"""
Launch a Slurm array to run CLAHE over a big EF-SEM TIFF.

Each array task processes a block of Z-slices by calling clahe_process.py
with --z-start / --z-end.
"""

import argparse
import math
import tempfile
import subprocess
from pathlib import Path

import tifffile


def main():
    parser = argparse.ArgumentParser(
        description="Submit a Slurm array job to run CLAHE on a BigTIFF."
    )
    parser.add_argument("-i", "--input", required=True, help="Input BigTIFF")
    parser.add_argument("-d", "--output-dir", required=True, help="Output directory for slices")
    parser.add_argument("--slices-per-job", type=int, default=10,
                        help="Number of Z-slices per array task (default: 10)")
    # You can add more knobs (partition, time, mem) if you like
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine Z from the TIFF
    with tifffile.TiffFile(str(input_path)) as tif:
        s = tif.series[0]
        Z = s.shape[0]

    slices_per_job = max(1, args.slices_per_job)
    num_jobs = math.ceil(Z / slices_per_job)

    print(f"Input:           {input_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Total Z slices:  {Z}")
    print(f"Slices per job:  {slices_per_job}")
    print(f"Array tasks:     0-{num_jobs-1}")

    # Build the Slurm script string
    script = f"""#!/bin/bash
#SBATCH --job-name=CLAHE_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --partition=ncpu   # <-- change to your real partition if needed
#SBATCH --array=0-{num_jobs-1}
#SBATCH --output=slurm-clahe-%A_%a.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

INPUT_TIF="{input_path}"
OUTPUT_DIR="{output_dir}"
TOTAL_Z={Z}
Z_PER_JOB={slices_per_job}

TASK_ID=${{SLURM_ARRAY_TASK_ID}}

# Compute Z range for this task
Z_START=$(( TASK_ID * Z_PER_JOB ))
Z_END=$(( Z_START + Z_PER_JOB - 1 ))
if [ $Z_END -ge $TOTAL_Z ]; then
    Z_END=$(( TOTAL_Z - 1 ))
fi

echo "Task $TASK_ID processing Z = $Z_START to $Z_END (TOTAL_Z={Z})"
python3 clahe_process.py \\
    -i "$INPUT_TIF" \\
    -d "$OUTPUT_DIR" \\
    --z-start $Z_START \\
    --z-end $Z_END
"""

    # Write to a temporary file and submit with sbatch
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    print(f"\nSubmitting array job with sbatch using script: {tmp_path}")
    result = subprocess.run(["sbatch", tmp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print("sbatch stdout:", result.stdout.strip())
    if result.stderr.strip():
        print("sbatch stderr:", result.stderr.strip())


if __name__ == "__main__":
    main()
