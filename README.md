[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/9FxAlQXs)

                                        |  Author       | Supriya Jayaraj |
                                        | ------------- | -------------   |
                                        | Matrikelnummer|    5452793      |

# Stock Price Prediction and Translation Tool

## Overview

This repository comprises two distinct projects: a comprehensive stock price prediction and trading strategy implementation using machine learning models, and a file translation tool for translating text content from various file formats.

### Stock Price Prediction and Trading Strategy

The stock price prediction project uses financial data, sentiment analysis of news, and technical indicators to make predictions for selected companies. The machine learning model is evaluated against historical stock prices, and a trading strategy is implemented to assess its effectiveness.

### File Translation Tool

The file translation tool is a Python-based utility that allows users to translate text content from various file formats. It utilizes Google Translate API for language translation and supports popular file types such as text files (.txt), Word documents (.docx), Excel spreadsheets (.xlsx), CSV files (.csv).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure (Stock Prediction)](#folder-structure-stock-prediction)
- [Folder Structure (File Translation)](#folder-structure-file-translation)
- [Usage with Jupyter Notebook](#usage-with-jupyter-notebook)
- [Output (Stock Prediction)](#output-stock-prediction)
- [Output (File Translation)](#output-file-translation)
- [License](#license)
- [Contact](#contact)

## Requirements

### Stock Prediction

- Python 3.x
- Libraries: numpy, pandas, seaborn, textblob, yfinance, ta, matplotlib, torch, cvxpy, sklearn

### File Translation

- Python 3.x
- Libraries: os, tkinter, langdetect, googletrans, docx, openpyxl, shutil, logging, csv, PyPDF2, reportlab

## Installation

### Stock Prediction

Ensure that Python 3.x is installed on your system. Use the following commands to set up the required environment:

```bash
sudo apt-get install python3 python3-pip
pip install -r stock_price_prediction/requirements.txt
```

### File Translation

Ensure that Python 3.x is installed on your system. Use the following commands to set up the required environment:

```bash
sudo apt-get install python3 python3-pip
pip install -r File_Translation_Tool/requirements.txt
```

## Usage

1. **Clone the repository:**

    ```bash
    git clone <link>
    ```

2. **Navigate to the project folder:**

    ```bash
    cd stock_price_prediction
    ```

    or

    ```bash
    cd File_Translation_Tool
    ```
    or

    ```bash
    cd Stock_price_prediction_notebook
    ```
3. **Run the code using a Python environment:**

    ```bash
    python main.py
    ```
    or

    ```bash
    python file_translation.py
    ```

### Folder Structure (Stock Prediction)

- `images/`: Stores output visualizations and graphs.
- `models/`: Contains the machine learning model code.
- `utils/`: Utility functions used in the project.
- `main.py`: Main script for executing the stock price prediction and trading strategy.

### Folder Structure (File Translation)

- `translation.log`: Log file for recording translation activities.
- `file_translator.py`: Main script for executing file translation.
- `requirements.txt`: List of required Python packages.
- `readme.txt`: Stores output visualizations and graphs (if applicable).
- `sample_output/`: Stores sample outputs expected from the code.

## Usage with Jupyter Notebook

1. **For a local installation, make sure you have pip installed and run:**

    ```bash
    pip install notebook
    ```

2. **Usage - Running Jupyter notebook:**

   - **Running in a local installation:**
     
      Launch with:
    
      ```bash
      jupyter notebook
      ```

   - **Running in a remote installation:**
   
      You need some configuration before starting Jupyter notebook remotely. See Running a notebook server.

   - **Ensure that you have input data i.e., stock prices and financial information in the proper format.**
   
   - **Add stocks or modify model architecture to suit your needs.**
   
   - **Execute the code and analyze results.**

## Output (Stock Prediction)

Predicted stock prices and trading strategy visualizations are saved as PNG files in the `images/` folder.

**Note:**
Please be aware that the visualized data may differ from the images present in the `images/` directory, as the end date is set to the current date. Variations in the data are expected, reflecting real-time updates.

## Output (File Translation)

Translated files are saved based on the file type and translation activities in the same folder.
Sample outputs are present in `sample_output/`

**Note:**
The current implementation lacks the capability to preserve the original file format during translation. Additional development efforts are required to address this limitation and ensure the output format matches the input format accurately. It's essential to acknowledge that the existing code successfully handles some `.docx` files, but further refinements are necessary for broader compatibility. This tool is currently in the prototype stage and may exhibit issues; ongoing work is planned to enhance its functionality.

**Important:** 
Please be aware that this tool is a prototype, and it may have limitations. The support for certain file formats, such as PDF, has been temporarily removed due to identified issues. Future updates to the code will aim to address these issues and improve overall functionality.

## Usage with Docker

### Stock Prediction Docker Image

1. **Pull the Docker image:**

    ```bash
    docker pull supu18/project-supu18
    ```

3. **Access Jupyter Notebook:**

    Open your web browser and navigate to `http://localhost:8888` to access the Jupyter Notebook for stock prediction.
    ```bash
       docker run -p 8888:8888 stocknotebook
    ```

4. **Note:**

    If you face any issues while running docker image, please directly run the application with the python commands mentioned.


## License

Both projects are distributed under the [MIT License](LICENSE).

## Contributions

You are welcome to contribute to these projects by opening an issue regarding any problems that you encounter or suggestions for improvement. Submit a pull request.

## Acknowledgements

Tutorials, online resources, and open-source projects related to stock analysis and prediction have influenced the stock prediction project. Special thanks go out to authors and contributors of these resources.

