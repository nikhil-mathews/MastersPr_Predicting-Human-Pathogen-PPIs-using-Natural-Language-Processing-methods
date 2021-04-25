## Masters Project: Predicting Human-Pathogen Protein-Protein Interactions using Natural Language Processing methods
We use multiple Natural Language Processing (NLP) methods available in deep learning and apply them to predict the interaction of proteins between Humans and Yersinia pestis by examining their respective amino acid sequences. Without using any biological knowledge, a model is developed that  gives a cross validation AUC score of 0.91 and an independent test score of 0.92, which rivals the reference research paper that uses amino acid  sequence and network data as well as  extensive use of bio-chemical properties, both sequential and network related, to make their predictions. The same model gave a score of 0.94 for predicting Humans-Virus PPI. This is done by combining advanced tools in neural machine translation into an integrated end-to-end deep learning framework as well as methods of preprocessing that are novel to the field of bioinformatics.

One way is to  account  for  the  possibility  that  common  sequence bits or “words” from both species play a role in prediction of interaction and this can be expressed with the help of *combine* configuration.

![combine](https://user-images.githubusercontent.com/52326197/116008184-3deb7d80-a5e1-11eb-90b6-20ce909563db.png)

We also employ a “3X” preprocessing configuration into an end to end framework to account for the possibilty that amino acids sequences from the left, right and center play a role in determining interaction.

![3X](https://user-images.githubusercontent.com/52326197/116008202-4d6ac680-a5e1-11eb-94b6-ffde0e9b8ab5.png)

The performance of Bi-LSTMs and CNN and compared and contrasted with multiple configurations and the best one chosen for the final model.
Use of Attention layers and Transformers are used and *differential join* configurations for Transformers are explored to give a better result.

The Final Model combines “3X CNN” configuration to address the separate nature of the two species and a “3X Transformers” configuration to  account  for  the  possibility  that  common  sequence bits from both species play a role in prediction of interaction. This is found to give competetive results for both Humans - Yersinia PPI as well as Humans - Virus PPI


# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
