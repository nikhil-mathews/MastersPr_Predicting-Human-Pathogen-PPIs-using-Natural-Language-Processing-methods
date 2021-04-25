## Masters Project: Predicting Human-Pathogen Protein-Protein Interactions using Natural Language Processing methods
We use multiple Natural Language Processing (NLP) methods available in deep learning and apply them to predict the interaction of proteins between Humans and Yersinia pestis by examining their respective amino acid sequences. Without using any biological knowledge, a model is developed that  gives a cross validation AUC score of 0.91 and an independent test score of 0.92, which rivals the reference research paper that uses amino acid  sequence and network data as well as  extensive use of bio-chemical properties, both sequential and network related, to make their predictions. The same model gave a score of 0.94 for predicting Humans-Virus PPI. This is done by combining advanced tools in neural machine translation into an integrated end-to-end deep learning framework as well as methods of preprocessing that are novel to the field of bioinformatics.


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
