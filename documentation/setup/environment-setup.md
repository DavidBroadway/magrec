`magrec` folder is a module that can be installed with `pip` in editable mode. This allows you to make changes to the code and have them reflected in the installed module without having to reinstall it. 

Run this from the `magrec` folder to install the module in editable mode:

`pip install --no-build-isolation --no-deps -e .`

```
| magrec                <--- root folder
├── documentation
├── logs
│  └── lightning_logs
├── magrec
│  ├── __init__.py
│  ├── misc
│  │  ├── data.py
│  │  ├── load.py
│  │  ├── plot.py
│  │  └── plotly_plot.py
│  ├── nn
│  │  ├── arch.py
│  │  └── models.py
│  ├── notebooks
│  │  ├── __init__.py
│  │  ├── Example.ipynb
│  │  └── test_current_propagation.ipynb
│  └── prop
│     ├── __init__.py
│     ├── conditions.py
│     ├── constants.py
│     ├── Filtering.py
│     ├── Fourier.py
│     ├── Kernel.py
│     ├── Pipeline.py
│     ├── Propagator.py
│     └── utils.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│  ├── current_reconstruction.py
│  └── profile.py
├── setup.cfg
└── setup.py
```
