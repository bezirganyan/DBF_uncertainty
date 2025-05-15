# DBF: Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion
## Description

The soure code for AISTATS 2025 paper: **[Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion](https://proceedings.mlr.press/v258/bezirganyan25a.html)**

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
Some code were borrowed from: https://github.com/jiajunsi/RCML, which is cited within our related work

## Citation
If you used the approach please cite our paper with:

> Bezirganyan, G., Sellami, S., Berti-Équille, L. &amp; Fournier, S.. (2025). Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion. <i>Proceedings of The 28th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 258:3142-3150 Available from https://proceedings.mlr.press/v258/bezirganyan25a.html.

or if you use latex with bibtex
```
@InProceedings{pmlr-v258-bezirganyan25a,
  title = 	 {Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion},
  author =       {Bezirganyan, Grigor and Sellami, Sana and Berti-Equille, Laure and Fournier, S{\'e}bastien},
  booktitle = 	 {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3142--3150},
  year = 	 {2025},
  editor = 	 {Li, Yingzhen and Mandt, Stephan and Agrawal, Shipra and Khan, Emtiyaz},
  volume = 	 {258},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {03--05 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v258/main/assets/bezirganyan25a/bezirganyan25a.pdf},
  url = 	 {https://proceedings.mlr.press/v258/bezirganyan25a.html},
```

## License

This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.
