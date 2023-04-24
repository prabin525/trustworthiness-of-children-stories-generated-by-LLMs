for model in 'opt' 'llama'
do
    sbatch  --job-name=story.$model template.slurm $model;
done;