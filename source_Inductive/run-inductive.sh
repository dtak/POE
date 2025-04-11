python collect-datasets.py

python main-inductive-images.py --dataset imagenet2000-seed0
python main-inductive-images.py --dataset imagenet2000-seed1
python main-inductive-images.py --dataset imagenet2000-seed2
python main-inductive-images.py --dataset imagenet2000-seed3
python main-inductive-images.py --dataset imagenet2000-seed4

for seed in {0..4}
do
    python main-inductive-functions.py --function cubed --seed $seed
    python main-inductive-functions.py --function sine --seed $seed
    python main-inductive-functions.py --function quasi --seed $seed
    python main-inductive-functions.py --function exp_quasi --seed $seed
    python main-inductive-functions.py --function exponential --seed $seed
done