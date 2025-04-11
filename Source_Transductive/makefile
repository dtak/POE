run:
	python submit_batch.py
	
morf:
	python morf.py

compare:
	python comparision_plots.py

debug:
	python -m ipdb -c continue submit_batch.py

vis:
	python vis.py

vis_debug:
	python -m ipdb -c continue vis.py

clear:
	rm -rf output/*

env:
	pip list --format=freeze > requirements.txt 


mofae: 
	python org.py

vis_mofae:
	python plt_mofae.py
