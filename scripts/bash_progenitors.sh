NAME="bolshoi_trees"
JOBNAME="job_$NAME"
TIME=02:00
CPUS=8
MEM=8GB
./bin/remote.py --cmd "./bin/get_progenitor_file.py --cpus $CPUS" --jobname $JOBNAME --time $TIME --cpus-per-task $CPUS --mem-per-cpu $MEM
