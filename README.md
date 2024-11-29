# DBAF: Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion
## Description

The soure code for paper: Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion

**The updated figures for the rebuttal of the paper are in the [updated_figures](./updated_figures) folder of this repository.**

## Requirements

```
pip install -r requirements.txt
```

## Usage

To run the script, use the following command:

```sh
python main.py [options]
```

### Arguments

The script accepts the following command-line arguments:

- `--batch-size` (int, default: 200): Input batch size for training.
- `--epochs` (int, default: 1000): Number of epochs to train the model.
- `--annealing_step` (int, default: 50): Gradually increase the value of lambda from 0 to 1.
- `--lr` (float, default: 0.003): Learning rate for the optimizer.
- `--agg` (str, default: 'conf_agg'): Aggregation method.
- `--runs` (int, default: 1): Number of runs for with different random seeds.  
- `--flambda` (float, default: 1): Lambda value for controlling the strictness of discounting
- `--activation` (str, default: 'softplus'): Activation function to be used.

To generrate the plots and tables, please run:
```sh
python produce_plots_and_tables.py
```

## Examples

Run the script with default parameters:

```sh
python script.py
```

Run the script with a custom batch size and learning rate:

```sh
python script.py --batch-size 100 --lr 0.001
```

## Thanks
Some code snippets were borrowed from: https://github.com/jiajunsi/RCML, which is within our related work

## License

This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.
