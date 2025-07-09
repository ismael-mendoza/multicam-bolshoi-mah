NAME="m12"
OUTPUT="bolshoi_${NAME}"
JOBNAME="job_${NAME}"
M_LOW=12.0
M_HIGH=12.2
N=10000

# To be safe, we should use masses > 12.0 from now on.
# M_LOW,M_HIGH = (12.0,12.2), (13, 14)

################# run pipeline #########################
./multicam/pipeline.py --outdir $OUTPUT make-ids --m-low $M_LOW --m-high $M_HIGH --n-haloes $N
./multicam/pipeline.py --outdir $OUTPUT make-dmcat
./multicam/pipeline.py --outdir $OUTPUT make-progenitors
./multicam/pipeline.py --outdir $OUTPUT make-subhaloes
./multicam/pipeline.py --outdir $OUTPUT combine-all
