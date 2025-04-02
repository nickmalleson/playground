# Schelling LLM 

Experiments with the Schelling model using LLMs/embeddings.

Adapted fromt he Schelling mesa example.
 - [Docs](https://mesa.readthedocs.io/latest/examples/basic/schelling.html).
 - [Code](https://github.com/projectmesa/mesa-schelling-example)

## Install

```
conda env create -f schelling_llm.yml
```

May need to install mesa separately as I'm not sure if conda has the latest version:

```
pip install -U "mesa[rec]"
```


## Run

Run the app interactively with `solara run app.py`

Or [analysis.ipynb](./analysis.ipynb) for more detailed analysis.
