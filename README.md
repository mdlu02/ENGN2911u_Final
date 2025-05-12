Public Github Repo: https://github.com/mdlu02/ENGN2911u_Final

All modified/added code is found in workspace/example_designs

workspace/example_designs/make_layer_shapes.ipynb contains a some code that allows for automatic parsing of MLP/CNN-based models found within Pytorch. While the code can also be applied to Transformer based models, there are differences in how our script parses attention heads compared to the ViT example that was given to us.

workspace/example_designs/analysis.ipynb contains code that we used to calculate summary statistics of our simulations and generate associated roof line plots of our data. This notebook can be used to visualize any Timeloop output.

workspace/example_designs/generate_results.py is a script for running a set of Timeloop parameter configs on a given accelerator design. The script handles configuring associated Timeloop .yaml files, running them, parsing the output text files into .csv files, and saving them to an appropriate location to avoid overwriting.