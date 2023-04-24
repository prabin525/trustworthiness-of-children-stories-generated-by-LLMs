for model in 'opt' 'llama'
do
    sbatch  --job-name=story.$modele template.slurm $model;
done;