# Optimizing for Interpretability Properties


## Instructions
* Please refer to Transductive to run Transductive Experiments 
* Please refer to Inductive to run Inductive Experiments 

**Run experiments in Transductive Setting:**
* Execute:
  * `cd auto`
  * `make`
* Then it will start writing the results into a directory called `output`
* Under output, every method has its own subdirectory e.g smoothgrad, lime, etc. 

**Visualize results:** `make vis`

**Creating an environment:** `micromamba create -n opt -c conda-forge python=3.10 -y`

**Activating:** `micromamba activate opt`

**Deactivate:** `micromamba deactivate`

**Install:** `pip install PACKAGE-NAME`


**After a set of experiments in the Tranductive Settings are complete, do:**
1. Look at the out files: `cat output/*/out*`. Does every experiment looks like it finished?
2. Look at the err files: `cat output/*/err*`. Are they empty? If so, no errors occured.
3. If everything ran smoothly, delete the out and err files:
  * `rm output/*/err*`
  * `rm output/*/out*`
