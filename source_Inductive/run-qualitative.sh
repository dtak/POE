python collect-datasets.py

for external in 0 0.0001 0.001 0.01 0.1
do
	for internal in 0 0.0001 0.01 10 1000
	do
		echo "external = $external, internal = $internal"
		python main-qualitative.py --external $external --internal $internal --dataset perturbations-sample0
		python main-qualitative.py --external $external --internal $internal --dataset perturbations-sample2
	done
done

python main-qualitative-smoothgrad.py --dataset "perturbations-sample0-delta(0.01)"
python main-qualitative-smoothgrad.py --dataset "perturbations-sample0-delta(0.05)"
python main-qualitative-smoothgrad.py --dataset "perturbations-sample0-delta(0.25)"
python main-qualitative-smoothgrad.py --dataset "perturbations-sample0-delta(1.25)"

python main-qualitative-smoothgrad.py --dataset "perturbations-sample2-delta(0.01)"
python main-qualitative-smoothgrad.py --dataset "perturbations-sample2-delta(0.05)"
python main-qualitative-smoothgrad.py --dataset "perturbations-sample2-delta(0.25)"
python main-qualitative-smoothgrad.py --dataset "perturbations-sample2-delta(1.25)"
